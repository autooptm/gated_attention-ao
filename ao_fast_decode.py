import contextlib
import os
import sys

import torch

_ENABLED = os.environ.get('GATED_ATTENTION_FAST', '1') != '0'
_TOLERANCE = 1e-2


def _warn(message):
    sys.stderr.write('[gated_attention] fast decode disabled: %s\n' % message)
    sys.stderr.flush()


def _ensure_opt_model_paths(model, patch):
    import torch.nn.functional as F

    norms = [m for m in model.modules() if m.__class__.__name__ == 'Qwen3RMSNorm']
    if norms and hasattr(F, 'rms_norm'):
        cls = type(norms[0])
        if 'rms_norm' not in cls.forward.__code__.co_names:
            def rms_forward(self, hidden_states):
                return F.rms_norm(hidden_states, (hidden_states.shape[-1],),
                                  self.weight, self.variance_epsilon)
            patch(cls, 'forward', rms_forward)

    rotary = [m for m in model.modules()
              if m.__class__.__name__ == 'Qwen3RotaryEmbedding'
              and getattr(m, 'inv_freq', None) is not None]
    if rotary and 'dynamic' not in str(getattr(rotary[0], 'rope_type', 'default')):
        cls = type(rotary[0])
        if '_ao_rope_table' not in cls.forward.__code__.co_names:
            tables = {}

            def rope_forward(self, x, position_ids):
                key = (str(x.device), x.dtype)
                entry = tables.get(key)
                if entry is None:
                    length = min(int(getattr(self, 'max_seq_len_cached', 0)) or 32768,
                                 32768)
                    pos = torch.arange(length, device=x.device, dtype=torch.float32)
                    freqs = torch.outer(pos, self.inv_freq.to(x.device).float())
                    emb = torch.cat((freqs, freqs), dim=-1)
                    entry = ((emb.cos() * self.attention_scaling).to(x.dtype),
                             (emb.sin() * self.attention_scaling).to(x.dtype))
                    tables[key] = entry
                cos_table, sin_table = entry
                flat = position_ids.reshape(-1)
                shape = (*position_ids.shape, cos_table.shape[-1])
                return cos_table[flat].view(shape), sin_table[flat].view(shape)

            patch(cls, 'forward', rope_forward)


class _FastDecode:

    def __init__(self, model):
        self.model = model
        self.cls = type(model)
        self.undo = []
        self.pool = {}
        self.step_states = {}
        self.disabled = False

    def _patch(self, owner, attr, value):
        self.undo.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, value)

    def __enter__(self):
        if not _ENABLED:
            return self
        try:
            self._install()
        except Exception as exc:                                  # noqa: BLE001
            self.disabled = True
            self._restore()
            _warn('%s: %s' % (type(exc).__name__, exc))
        return self

    def __exit__(self, *exc_info):
        self._restore()
        return False

    def remove(self):
        self._restore()

    def _restore(self):
        while self.undo:
            owner, attr, original = self.undo.pop()
            setattr(owner, attr, original)
        self.step_states.clear()
        self.pool.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _install(self):
        from transformers.cache_utils import StaticCache, StaticCacheConfig
        from transformers.generation.utils import GenerationMixin

        if not torch.cuda.is_available():
            raise RuntimeError('no CUDA device')

        config_init = StaticCacheConfig.__init__

        def patched_config_init(cfg, batch_size=1, max_cache_len=1, *args, **kwargs):
            config_init(cfg, batch_size, max_cache_len, *args, **kwargs)

        self._patch(StaticCacheConfig, '__init__', patched_config_init)

        original_get_cache = GenerationMixin._get_cache
        pool, model = self.pool, self.model

        def get_cache(mixin, cache_implementation, batch_size, max_cache_len, device,
                      *args, **kwargs):
            if cache_implementation != 'static':
                return original_get_cache(mixin, cache_implementation, batch_size,
                                          max_cache_len, device, *args, **kwargs)
            key = (int(batch_size), int(max_cache_len))
            cache = pool.get(key)
            if cache is None:
                try:
                    cache = StaticCache(config=mixin.config, batch_size=key[0],
                                        max_cache_len=key[1], device=device,
                                        dtype=model.dtype)
                except TypeError:
                    cache = StaticCache(config=mixin.config, max_batch_size=key[0],
                                        max_cache_len=key[1], device=device,
                                        dtype=model.dtype)
                pool[key] = cache
            else:
                cache.reset()
            mixin._cache = cache
            return cache

        self._patch(GenerationMixin, '_get_cache', get_cache)

        original_prepare = self.cls.prepare_inputs_for_generation

        def prepare(module, input_ids, **kwargs):
            out = original_prepare(module, input_ids, **kwargs)
            given = kwargs.get('attention_mask')
            got = out.get('attention_mask')
            if given is not None and given.dim() == 2 and got is not None and got.dim() == 4:
                out['attention_mask'] = given
            return out

        self._patch(self.cls, 'prepare_inputs_for_generation', prepare)

        original_unfinished = GenerationMixin._has_unfinished_sequences

        def has_unfinished(mixin, this_peer_finished, synced_gpus, device,
                           cur_len=None, max_length=None):
            if not synced_gpus and cur_len is not None and max_length is not None:
                return cur_len < max_length
            return original_unfinished(mixin, this_peer_finished, synced_gpus, device,
                                       cur_len=cur_len, max_length=max_length)

        self._patch(GenerationMixin, '_has_unfinished_sequences', has_unfinished)
        _ensure_opt_model_paths(self.model, self._patch)
        self._patch(self.cls, 'forward', self._make_forward())

    def _make_forward(self):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        original_forward = self.cls.forward
        state_of = self.step_states
        owner = self

        def forward(module, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, labels=None,
                    use_cache=None, output_attentions=None, output_hidden_states=None,
                    return_dict=None, cache_position=None, num_logits_to_keep=0,
                    **kwargs):

            def eager():
                return original_forward(
                    module, input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states, return_dict=return_dict,
                    cache_position=cache_position,
                    num_logits_to_keep=num_logits_to_keep, **kwargs)

            if (owner.disabled
                    or input_ids is None or input_ids.dim() != 2
                    or input_ids.shape[1] != 1
                    or inputs_embeds is not None or labels is not None
                    or output_attentions or output_hidden_states
                    or past_key_values is None
                    or type(past_key_values).__name__ != 'StaticCache'
                    or cache_position is None or int(num_logits_to_keep) != 1):
                return eager()

            batch = int(input_ids.shape[0])
            length = int(past_key_values.get_max_cache_shape())
            key = (batch, length, id(past_key_values))
            state = state_of.get(key)

            if state is None:
                try:
                    state = owner._build_step(module, original_forward, input_ids,
                                           attention_mask, past_key_values,
                                           cache_position, key)
                except Exception as exc:                          # noqa: BLE001
                    owner.disabled = True
                    _warn('%s: %s' % (type(exc).__name__, exc))
                    return eager()
                state_of[key] = state
            else:
                state['ids'].copy_(input_ids)
                state['pos'].copy_(cache_position)
                state['g'].replay()

            return CausalLMOutputWithPast(logits=state['logits'],
                                          past_key_values=past_key_values)

        return forward

    def _build_step(self, module, original_forward, input_ids, attention_mask,
                 cache, cache_position, key):
        batch, length = key[0], key[1]

        if attention_mask is not None:
            filled = int(cache_position.max().item()) + 1
            if attention_mask.dim() == 4:
                clean = bool((attention_mask[..., :filled] == 0).all())
            else:
                clean = bool(attention_mask[:, :filled].all())
            if not clean:
                raise RuntimeError('the batch is left-padded, so the decode mask '
                                   'cannot be replaced by a constant one')

        ids = input_ids.clone()
        pos = cache_position.clone()
        mask = torch.ones((batch, length), dtype=torch.long, device=input_ids.device)

        def once():
            return original_forward(module, input_ids=ids, attention_mask=mask,
                                    past_key_values=cache, use_cache=True,
                                    cache_position=pos, num_logits_to_keep=1,
                                    return_dict=True)

        reference = once().logits.detach().clone()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                once()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = once()
        g.replay()

        scale = float(reference.abs().max()) or 1.0
        drift = float((out.logits.float() - reference.float()).abs().max())
        if drift > _TOLERANCE * scale:
            raise RuntimeError('the optimized step disagrees with the stock forward '
                               '(max delta %.3g against a scale of %.3g)' % (drift, scale))

        return {'ids': ids, 'pos': pos, 'mask': mask, 'g': g,
                'logits': out.logits}


_ACTIVE = []


def install(model):
    helper = _FastDecode(model)
    helper.__enter__()
    _ACTIVE.append(helper)
    return helper


@contextlib.contextmanager
def fast_decode(model):
    helper = _FastDecode(model)
    with helper:
        yield helper


def cache_implementation():
    if not _ENABLED:
        return None
    if not _ACTIVE or _ACTIVE[-1].disabled or not _ACTIVE[-1].undo:
        return None
    return 'static'
