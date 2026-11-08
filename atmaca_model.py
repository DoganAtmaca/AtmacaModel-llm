import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AtmacaConfig:
    vocab_size: int = 16384
    context_length: int = 512
    hidden_size: int = 512
    num_hidden_layers: int = 10
    num_attention_heads: int = 8
    rms_norm_eps: float = 1e-6
    intermediate_size: int = 1408 
    rope_theta: float = 10000.0
    dropout: float = 0.1

    @classmethod
    def from_preset(cls, preset: str = "35m", vocab_size: int = 16384) -> "AtmacaConfig":
        """RTX 4070 Ti 12GB için hazır model ölçekleri:
        - '25m': Küçük model (~25.2M parametre, 8 katman, 384 boyut)
        - '35m': Varsayılan önerilen model (~34.1M parametre, 8 katman, 512 boyut — 50MB temiz korpus için en dengeli ölçek)
        - '40m': Orta model (~40.5M parametre, 10 katman, 512 boyut)
        - '60m': Büyük model (~57.5M parametre, 12 katman, 576 boyut — daha büyük korpuslar için)
        """
        p = (preset or "35m").lower().strip()
        if p == "25m":
            return cls(
                vocab_size=vocab_size,
                context_length=512,
                hidden_size=384,
                num_hidden_layers=8,
                num_attention_heads=6,
                intermediate_size=1536,
                dropout=0.1,
            )
        if p == "35m":
            return cls(
                vocab_size=vocab_size,
                context_length=512,
                hidden_size=512,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=1408,
                dropout=0.1,
            )
        if p == "60m":
            return cls(
                vocab_size=vocab_size,
                context_length=512,
                hidden_size=576,
                num_hidden_layers=12,
                num_attention_heads=9,
                intermediate_size=1536,
                dropout=0.1,
            )
        return cls(vocab_size=vocab_size)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AtmacaConfig":
        gecerli = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**gecerli)


class AtmacaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))  # hidden_size (ör. 384) tane 1'lerden oluşan ölçek vektörü

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # dim=-1 yön belirtiyor: 3 boyutlu tensörün en son (hidden_size) boyutuna normalizasyon yapılır.
        # Her kelime vektörü (384'lük) yalnızca kendi içinde normalize edilir; diğer cümlelerden etkilenmez.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        norm_x = torch.mean(hidden_states ** 2, dim=-1, keepdim=True)
        normed_x = hidden_states * torch.rsqrt(norm_x + self.eps)
        return self.weight * normed_x.to(input_dtype)


# Bu kısım RoPE (Rotary Positional Embedding) mekanizmasını uygular.
class AtmacaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 512, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len_cached = 0

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._tablo_guncelle(max_position_embeddings, device=device)

    def _tablo_guncelle(self, seq_len: int, device=None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device if device is not None else self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        toplam_uzunluk = offset + seq_len
        if toplam_uzunluk > self.max_seq_len_cached or self.cos_cached.device != x.device:
            self._tablo_guncelle(max(toplam_uzunluk, self.max_seq_len_cached), device=x.device)

        cos = self.cos_cached[offset:toplam_uzunluk].to(dtype=x.dtype)
        sin = self.sin_cached[offset:toplam_uzunluk].to(dtype=x.dtype)
        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Formüldeki (X*Cos - Y*Sin) eksisini uygulayabilmek için,
    # matrisin ikinci yarısını eksi ile çarpıp ilk yarısıyla yer değiştiriyoruz.
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)

    q_b1, q_b2 = q.chunk(2, dim=-1)
    q_rotated = torch.cat((-q_b2, q_b1), dim=-1)

    k_b1, k_b2 = k.chunk(2, dim=-1)
    k_rotated = torch.cat((-k_b2, k_b1), dim=-1)

    q_new = (q * cos) + (q_rotated * sin)
    k_new = (k * cos) + (k_rotated * sin)

    return q_new, k_new


class AtmacaAttention(nn.Module):
    def __init__(self, config: AtmacaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}), num_attention_heads ({self.num_heads}) değerine tam bölünmelidir!")

        # Her bir kafanın boyutu (384 / 6 = 64)
        self.head_dim = self.hidden_size // self.num_heads

        # Gelen 384 boyutlu kelimeyi alır, Soru (Q), Anahtar (K) ve Değer (V) matrislerine çevirirler.
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        # Farklı kafalardan çıkan sonuçları tekrar tek parça 384'lük vektöre dönüştürür.
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len, _ = hidden_states.size()
        cos, sin = position_embeddings

        # 1. KELİMELERİ Q, K, V'YE ÇEVİRME
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. ÇOKLU KAFA (MULTI-HEAD) İÇİN BÖLME VE ŞEKİLLENDİRME -> [Batch, Head, Seq, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. ROPE (SAAT GİBİ DÖNDÜRME) İŞLEMİ (Dikkat! Sadece Q ve K döner, V dönmez)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        present_kv = (key_states, value_states) if use_cache else None
        toplam_k_len = key_states.size(2)

      
        if attention_mask is None and past_len == 0:
            
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=(seq_len > 1)
            )
        elif attention_mask is None and seq_len == 1:
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=False
            )
        else:
            if attention_mask is not None and attention_mask.dim() == 4:
                mask = attention_mask
            else:
                causal = torch.ones((seq_len, toplam_k_len), dtype=torch.bool, device=query_states.device).tril(diagonal=past_len)
                mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, toplam_k_len]
                if attention_mask is not None:
                    pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
                    mask = mask & pad_mask
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=mask, is_causal=False
            )

        # 5. KAFALARI GERİ BİRLEŞTİRME VE ÇIKIŞ
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        cikti = self.resid_dropout(self.o_proj(attn_output))

        if use_cache:
            return cikti, present_kv
        return cikti


class AtmacaMLP(nn.Module):
    def __init__(self, config: AtmacaConfig):
        super().__init__()
        # Orijinal boyut (384) ve Genişletilmiş Ara Boyut (1536)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # AKTİVASYON FONKSİYONU (SiLU)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kapi_filitresi = self.act_fn(self.gate_proj(hidden_states))
        genisletilmis_bilgi = self.up_proj(hidden_states)
        filtrelenmis_bilgi = kapi_filitresi * genisletilmis_bilgi
        return self.dropout(self.down_proj(filtrelenmis_bilgi))


class AtmacaDecoderLayer(nn.Module):
    def __init__(self, config: AtmacaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.input_layernorm = AtmacaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = AtmacaAttention(config)

        self.post_attention_layernorm = AtmacaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AtmacaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if use_cache:
            attn_out, present_kv = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=True,
            )
        else:
            attn_out = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=False,
            )
            present_kv = None

        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden_states, present_kv
        return hidden_states


def belge_izolasyon_maskesi(input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """512 tokenlık paketlenen pencerelerde `<|endoftext|>` sınırından önceki belgenin
    sonraki belgeye attention sızdırmasını engelleyen [B, 1, T, T] boolean causal maske oluşturur."""
    _, seq_len = input_ids.shape
    is_eos = (input_ids == eos_id).to(torch.int32)
    shifted = torch.zeros_like(is_eos)
    shifted[:, 1:] = is_eos[:, :-1]
    doc_ids = shifted.cumsum(dim=1)  # Her <|endoftext|> sonrası belge kimliği 1 artar
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # [B, T, T]
    causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device).tril()
    return (same_doc & causal.unsqueeze(0)).unsqueeze(1).contiguous()  # [B, 1, T, T]


class AtmacaModel(nn.Module):
    def __init__(self, config: AtmacaConfig):
        super().__init__()
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.rotary_emb = AtmacaRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.context_length,
            base=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [AtmacaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.norm = AtmacaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        eos_id: Optional[int] = None,
    ):
        if (
            eos_id is not None
            and attention_mask is None
            and not use_cache
            and past_key_values is None
            and input_ids.shape[1] > 1
            and (input_ids[:, :-1] == eos_id).any()
        ):
            attention_mask = belge_izolasyon_maskesi(input_ids, eos_id)

        hidden_states = self.embed_dropout(self.embed_tokens(input_ids))

        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        cos, sin = self.rotary_emb(hidden_states, seq_len=hidden_states.shape[1], offset=past_len)
        position_embeddings = (cos, sin)

        yeni_kv_listesi = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            katman_kv = past_key_values[idx] if past_key_values is not None else None
            if use_cache:
                hidden_states, p_kv = layer(
                    hidden_states,
                    position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=katman_kv,
                    use_cache=True,
                )
                yeni_kv_listesi.append(p_kv)
            else:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=katman_kv,
                    use_cache=False,
                )

        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, yeni_kv_listesi
        return hidden_states


class AtmacaForCausalLM(nn.Module):
    def __init__(self, config: AtmacaConfig):
        super().__init__()
        self.config = config

        self.model = AtmacaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_hidden_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        eos_id: Optional[int] = None,
    ):
        if use_cache:
            hidden_states, yeni_kv = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                eos_id=eos_id,
            )
            logits = self.lm_head(hidden_states)
            return logits, yeni_kv

        hidden_states = self.model(input_ids, attention_mask=attention_mask, eos_id=eos_id)
        logits = self.lm_head(hidden_states)
        return logits


def model_kaydet(dosya_yolu: str, model: AtmacaForCausalLM, config: AtmacaConfig, ekstra: Optional[dict] = None):
    """Hem ağırlıkları hem de AtmacaConfig bilgisini birlikte kaydeder."""
    paket = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }
    if ekstra:
        paket.update(ekstra)
    torch.save(paket, dosya_yolu)


def model_yukle(dosya_yolu: str, device: torch.device = torch.device("cpu")) -> Tuple[AtmacaForCausalLM, AtmacaConfig, dict]:
    """Hem eski saf state_dict (.pth) dosyalarını hem de yeni config'li checkpoint paketlerini sorunsuz yükler."""
    yuklenen = torch.load(dosya_yolu, map_location="cpu", weights_only=True)
    if isinstance(yuklenen, dict) and "model_state_dict" in yuklenen:
        config = AtmacaConfig.from_dict(yuklenen.get("config", {}))
        state_dict = yuklenen["model_state_dict"]
        meta = {k: v for k, v in yuklenen.items() if k != "model_state_dict"}
    else:
        # Geriye dönük uyumluluk: doğrudan state_dict kaydedilmiş eski .pth dosyası
        emb_w = yuklenen.get("model.embed_tokens.weight")
        if emb_w is not None:
            vocab_size, hidden_size = emb_w.shape
            n_layers = sum(1 for k in yuklenen if k.endswith("self_attn.q_proj.weight"))
            inter_size = yuklenen["model.layers.0.mlp.up_proj.weight"].shape[0]
            n_heads = max(1, hidden_size // 64)
            config = AtmacaConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                num_attention_heads=n_heads,
                intermediate_size=inter_size,
            )
        else:
            config = AtmacaConfig()
        state_dict = yuklenen
        meta = {}

    model = AtmacaForCausalLM(config)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, config, meta

if __name__ == "__main__":
    config = AtmacaConfig()
    model = AtmacaForCausalLM(config)

    print("================ MODEL MİMARİSİ ================")
    print(model)
    param_sayisi = sum(p.numel() for p in model.parameters())
    print(f"\nToplam Parametre Sayısı: {param_sayisi:,} (~{param_sayisi / 1e6:.2f}M)")

    print("\n================ SİMÜLASYON BAŞLIYOR ================")
    sahte_girdi = torch.randint(0, config.vocab_size, (8, config.context_length))
    print(f"Modele giren verinin şekli: {sahte_girdi.shape} -> (8 Paket, {config.context_length} Token)")

    ciktilar_logits = model(sahte_girdi)
    print(
        f"Modelden çıkan verinin şekli: {ciktilar_logits.shape} -> "
        f"(8 Paket, {config.context_length} Token, {config.vocab_size} Kelimelik Puan Tablosu)"
    )
    print("\nMimari başarıyla çalışıyor ve sıradaki kelimeleri tahmin etmeye hazır!")
