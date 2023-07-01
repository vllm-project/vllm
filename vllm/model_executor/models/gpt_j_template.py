try:
    @register_loader("GPTJConfig")
    class GPTJLoader(ModelLoader):
        @property
        def architecture_name(self):
            return "GPTJForCausalLM"

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

            self.set_decoder(
                spec.decoder,
                model.transformer,
                model.config.rotary_dim,
                model.config.n_head,
            )
            self.set_linear(spec.decoder.projection, model.lm_head)
            return spec

        def set_vocabulary(self, spec, tokens):
            spec.register_vocabulary(tokens)

        def set_config(self, config, model, tokenizer):
            config.bos_token = tokenizer.bos_token
            config.eos_token = tokenizer.eos_token
            config.unk_token = tokenizer.unk_token

        def set_decoder(self, spec, module, rotary_dim, num_heads):
            spec.scale_embeddings = False
            self.set_embeddings(spec.embeddings, module.wte)
            self.set_layer_norm(spec.layer_norm, module.ln_f)

            for layer_spec, layer in zip(spec.layer, module.h):
                self.set_layer_norm(layer_spec.shared_layer_norm, layer.ln_1)

                qw = layer.attn.q_proj.weight.numpy()
                kw = layer.attn.k_proj.weight.numpy()
                vw = layer.attn.v_proj.weight.numpy()

                qw = utils.permute_for_sliced_rotary(qw, num_heads, rotary_dim)
                kw = utils.permute_for_sliced_rotary(kw, num_heads, rotary_dim)

                layer_spec.self_attention.linear[0].weight = np.concatenate((qw, kw, vw))
                self.set_linear(layer_spec.self_attention.linear[1], layer.attn.out_proj)

                self.set_linear(layer_spec.ffn.linear_0, layer.mlp.fc_in)
                self.set_linear(layer_spec.ffn.linear_1, layer.mlp.fc_out)
except:
    pass