try:
    @register_loader("CodeGenConfig")
    class CodeGenLoader(ModelLoader):
        @property
        def architecture_name(self):
            return "CodeGenForCausalLM"

        def get_model_spec(self, model):
            spec = transformer_spec.TransformerDecoderModelSpec.from_config(
                model.config.n_layer,
                model.config.n_head,
                pre_norm=True,
                activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
                rotary_dim=model.config.rotary_dim,
                rotary_interleave=False,
                parallel_residual=True,
                shared_layer_norm=True,
            )

            mp_num = 4
            if hasattr(model.config, "head_dim") and model.config.head_dim in [128, 256]:
                # models forked from "Salesforce/codegen2-1B" and "Salesforce/codegen2-3_7B"
                # use a special setting of mp_num=8, all other using 4
                # these model.config's use a special setting of head_dim
                mp_num = 8

            self.set_decoder(
                spec.decoder,
                model.transformer,
                model.config.rotary_dim,
                model.config.n_head,
                model.config.n_embd,
                mp_num=mp_num,
            )
            self.set_linear(spec.decoder.projection, model.lm_head)
            return spec

        def set_vocabulary(self, spec, tokens):
            spec.register_vocabulary(tokens)

        def set_config(self, config, model, tokenizer):
            config.bos_token = tokenizer.bos_token
            config.eos_token = tokenizer.eos_token
            config.unk_token = tokenizer.unk_token

        def set_decoder(self, spec, module, rotary_dim, num_heads, embed_dim, mp_num):
            spec.scale_embeddings = False
            self.set_embeddings(spec.embeddings, module.wte)
            self.set_layer_norm(spec.layer_norm, module.ln_f)

            base_permutation = np.arange(0, mp_num * 3).reshape(-1, 3).T.flatten().tolist()
            local_dim = embed_dim // mp_num
            permutation = np.concatenate(
                [np.arange(i * local_dim, (i + 1) * local_dim) for i in base_permutation]
            )

            for layer_spec, layer in zip(spec.layer, module.h):
                self.set_layer_norm(layer_spec.shared_layer_norm, layer.ln_1)
                # [start convert CodeGen to GPT-J format]
                # numpy conversion, adapted from torch code in
                # see https://github.com/fauxpilot/fauxpilot/blob/fb4073a9078dd001ebeb7dfefb8cb2ecc8a88f4b/converter/codegen_gptj_convert.py # noqa
                qkv_proj = layer.attn.qkv_proj.weight.numpy()

                # GPT-J and CodeGen slice up the qkv projection slightly differently.
                # the following permutation brings Codegen 'qkv_proj'
                # in GPT-J order of qw, vw, kw
                # we permute the *rows* here because the computation is xA.T
                new_qkv_proj = qkv_proj[permutation, :]
                # the name QKV is misleading here; they are actually stored in QVK
                qw, vw, kw = np.array_split(new_qkv_proj, 3, axis=0)
                # [end convert CodeGen to GPT-J.]

                qw = utils.permute_for_sliced_rotary(qw, num_heads, rotary_dim)
                kw = utils.permute_for_sliced_rotary(kw, num_heads, rotary_dim)

                layer_spec.self_attention.linear[0].weight = np.concatenate((qw, kw, vw))
                self.set_linear(layer_spec.self_attention.linear[1], layer.attn.out_proj)

                self.set_linear(layer_spec.ffn.linear_0, layer.mlp.fc_in)
                self.set_linear(layer_spec.ffn.linear_1, layer.mlp.fc_out)
except:
    pass