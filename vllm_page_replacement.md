# vLLM 페이지 교체 알고리즘 쉽게 이해하기

---

## 먼저 알아야 할 배경 지식

### LLM 추론이란?

ChatGPT 같은 AI에게 질문을 보내면, AI는 질문을 읽고 답변을 한 단어(토큰)씩 생성합니다.
이때 AI는 이전에 읽은 내용을 기억하면서 다음 단어를 예측하는데, 이 "기억"을 **KV 캐시(Key-Value Cache)** 라고 합니다.

### KV 캐시란?

- AI가 텍스트를 처리할 때, 각 단어에 대한 중간 계산 결과(Key, Value)를 저장해두는 공간입니다.
- 이걸 저장해두면, 같은 내용을 다시 계산하지 않아도 돼서 속도가 빨라집니다.
- 문제는 GPU 메모리가 한정되어 있어서, **모든 대화의 KV 캐시를 동시에 저장할 수 없다**는 점입니다.

### vLLM이란?

vLLM은 LLM을 빠르고 효율적으로 서빙하는 오픈소스 라이브러리입니다.
여러 사용자의 요청을 동시에 처리할 때, KV 캐시를 블록(block) 단위로 나눠서 GPU 메모리를 효율적으로 관리합니다.
이 방식을 **PagedAttention** 이라고 합니다.

### 페이지 교체 알고리즘이 왜 필요한가?

GPU 메모리(KV 캐시 공간)는 한정적입니다. 새로운 요청이 들어왔는데 공간이 부족하면,
기존에 저장된 블록 중 하나를 **버려야(evict)** 합니다.

이때 **"어떤 블록을 버릴 것인가?"** 를 결정하는 규칙이 바로 **페이지 교체 알고리즘**입니다.

---

## vLLM의 페이지 교체 알고리즘

vLLM은 크게 두 가지 레이어에서 교체 알고리즘이 작동합니다.

```
[사용자 요청]
      |
      v
[GPU KV 캐시] <-- 1. LRU 알고리즘 사용
      |
   (공간 부족 시 CPU/디스크로 내보냄)
      |
      v
[CPU/디스크 오프로딩 캐시] <-- 2. LRU 또는 ARC 알고리즘 사용
```

---

## 1. GPU KV 캐시: LRU (Least Recently Used)

### LRU란?

**"가장 오랫동안 사용되지 않은 것을 먼저 버린다"** 는 전략입니다.

### 비유로 이해하기

책상 위에 책을 올려놓을 수 있는 공간이 5칸 있다고 가정합니다.

```
[책상 공간]
| 수학책 | 영어책 | 과학책 | 역사책 | 국어책 |
  (오래됨)                            (최근)
```

책을 꺼내서 볼 때마다 오른쪽 끝으로 옮깁니다. (최근 사용)
새 책을 올려놓아야 하는데 공간이 없으면, **왼쪽 끝(가장 오래된 것)** 을 치웁니다.

### vLLM에서의 구현

vLLM은 `FreeKVCacheBlockQueue`라는 **이중 연결 리스트(doubly linked list)** 로 LRU를 구현합니다.

```
[Head] <-> [블록A] <-> [블록B] <-> [블록C] <-> [Tail]
            (LRU)                   (MRU)
            가장 먼저                가장 최근
            버려질 블록              사용된 블록
```

- 새 블록이 필요하면 **Head 쪽(LRU)** 에서 꺼냅니다.
- 블록 사용이 끝나면 **Tail 쪽(MRU)** 에 붙입니다.
- 이중 연결 리스트이므로 중간 블록을 O(1) 시간에 제거할 수 있습니다.

### Prefix Caching과의 연동

vLLM은 **Prefix Caching** 기능도 지원합니다. 이는 여러 요청이 같은 앞부분(prefix)을 공유하면,
KV 캐시를 재활용하는 기능입니다.

예를 들어, 두 사용자가 같은 시스템 프롬프트를 쓴다면:

```
[시스템 프롬프트 블록] <- 두 요청이 공유 (evict 안 함)
[사용자A 대화 블록  ]
[사용자B 대화 블록  ]
```

- 캐시된 블록이 다른 요청에 재사용되면, `touch()` 함수로 **eviction 후보에서 제외**합니다.
- 아무도 사용하지 않는 블록만 LRU 순서에 따라 버려집니다.

### 동점 처리 규칙

같은 요청에서 동시에 해제되는 블록이 여러 개일 때는 추가 규칙이 있습니다:
**"해시 토큰이 많은 블록(체인의 꼬리 쪽)"을 더 먼저 버립니다.**

이는 prefix 재사용 가능성이 높은 앞쪽 블록을 최대한 살려두기 위한 전략입니다.

---

## 2. KV 오프로딩 캐시: LRU 또는 ARC

GPU 메모리에서 쫓겨난 블록은 **CPU 메모리나 디스크**로 내려갈 수 있습니다(오프로딩).
이 오프로딩 캐시에서도 공간 관리가 필요하며, 두 가지 알고리즘 중 선택할 수 있습니다.

---

### 2-1. LRU 오프로딩 (`LRUOffloadingManager`)

GPU 캐시와 동일한 LRU 방식을 CPU/디스크 레이어에도 적용합니다.
Python의 `OrderedDict`(순서가 있는 딕셔너리)로 구현되어 있습니다.

```python
# OrderedDict는 삽입/접근 순서를 기억합니다.
# move_to_end()로 최근 사용 항목을 뒤로 이동합니다.
blocks = OrderedDict()
blocks.move_to_end(block_hash)  # MRU 위치로 이동
```

단순하고 구현이 쉽지만, **재사용 빈도는 고려하지 않는다**는 단점이 있습니다.

---

### 2-2. ARC (`ARCOffloadingManager`)

**ARC(Adaptive Replacement Cache)** 는 LRU보다 한 단계 발전된 알고리즘입니다.

#### 핵심 아이디어

LRU는 "얼마나 최근에 썼나"만 봅니다.
ARC는 **"최근에 썼나" + "얼마나 자주 쓰나"** 를 모두 고려합니다.

#### 4개의 리스트

ARC는 내부적으로 4개의 리스트를 관리합니다:

| 리스트 | 설명 |
|--------|------|
| **T1** (Recent) | 딱 한 번 접근한 블록들 (신규 블록) |
| **T2** (Frequent) | 두 번 이상 접근한 블록들 (자주 쓰는 블록) |
| **B1** (Ghost T1) | T1에서 버려진 블록의 "흔적" (실제 데이터는 없음) |
| **B2** (Ghost T2) | T2에서 버려진 블록의 "흔적" (실제 데이터는 없음) |

```
실제 데이터가 있는 캐시:
  T1 [블록E, 블록F, 블록G]  <- 최근에 한 번 접근
  T2 [블록A, 블록B, 블록C]  <- 여러 번 접근한 인기 블록

흔적만 남은 Ghost 리스트:
  B1 [블록H의 흔적, 블록I의 흔적]  <- 예전에 T1에서 쫓겨남
  B2 [블록J의 흔적, 블록K의 흔적]  <- 예전에 T2에서 쫓겨남
```

#### 작동 방식

**블록 첫 접근 시:**
- T1에 추가합니다. (신규 블록)

**블록 재접근 시 (`touch`):**
- T1에 있으면 → **T2로 승격** (자주 쓰는 블록으로 인정)
- T2에 있으면 → T2 내에서 MRU 위치로 이동
- B1(흔적)에 있으면 → `target_t1_size`를 **늘립니다** (최근성 중요)
- B2(흔적)에 있으면 → `target_t1_size`를 **줄입니다** (빈도 중요)

**블록을 버려야 할 때:**
- T1 크기 >= `target_t1_size` → T1의 LRU 블록을 버리고 B1에 흔적 추가
- T1 크기 < `target_t1_size` → T2의 LRU 블록을 버리고 B2에 흔적 추가

#### 자기 학습(Adaptive) 메커니즘

```
상황: 최근에 한 번씩만 본 블록들이 자주 재요청됨
  -> B1 히트 증가 -> target_t1_size 증가 -> T1 공간 확대
  -> 최근에 본 블록을 더 오래 유지

상황: 자주 반복 접근하는 블록들이 있음
  -> B2 히트 증가 -> target_t1_size 감소 -> T2 공간 확대
  -> 인기 있는 블록을 더 오래 유지
```

이처럼 ARC는 워크로드 패턴에 따라 **자동으로 전략을 조정**합니다.

---

## LRU vs ARC 비교 정리

| 항목 | LRU | ARC |
|------|-----|-----|
| 고려 요소 | 최근 사용 시점 | 최근 사용 + 접근 빈도 |
| 구현 복잡도 | 단순 | 복잡 |
| 메모리 오버헤드 | 낮음 | 높음 (Ghost 리스트 때문) |
| 워크로드 적응력 | 없음 | 있음 (자동 조정) |
| 사용 위치 | GPU 캐시, CPU 오프로딩 | CPU 오프로딩 |

---

## 전체 흐름 요약

```
1. 사용자 요청 도착
        |
        v
2. GPU KV 캐시에 공간 있음?
   YES -> 블록 할당 후 처리
   NO  -> LRU 순서로 가장 오래된 블록 선택
        |
        v
3. 선택된 블록을 CPU/디스크로 오프로딩
   (LRU 또는 ARC로 관리)
        |
        v
4. 비워진 GPU 공간에 새 블록 할당
        |
        v
5. 나중에 오프로딩된 블록이 다시 필요하면
   GPU로 불러옴 (swap-in)
```

---

## GPU KV 캐시에 적용해볼 만한 대안 전략들

LLM 서빙의 KV 캐시는 일반적인 웹 캐시와 다른 특성을 가집니다.

### LLM KV 캐시의 특수한 특성

| 특성 | 설명 |
|------|------|
| **블록 체인 구조** | 블록들이 앞→뒤로 이어진 체인을 이룸 (앞 블록이 더 재사용 가치가 높음) |
| **Prefix 공유** | 시스템 프롬프트 등 수많은 요청이 동일한 앞부분 블록을 공유 |
| **스캔 오염** | 일회성 긴 요청이 캐시를 가득 채워 유용한 블록들을 밀어낼 수 있음 |
| **접근 시점 집중** | 블록은 요청이 처리되는 동안만 접근되고, 이후엔 재사용되거나 방치됨 |

이 특성들을 고려하면 LRU보다 더 적합한 전략들이 있습니다.

---

### 전략 1: ARC (Adaptive Replacement Cache) — GPU 캐시로 확장

#### 핵심 아이디어

ARC는 현재 vLLM에서 **CPU 오프로딩 레이어**에만 사용되고 있습니다.
이를 **GPU KV 캐시에도 직접 적용**하는 것이 첫 번째 대안입니다.

#### 작동 방식 (재요약)

```
T1 (한 번 접근한 블록)  +  T2 (여러 번 접근한 인기 블록)
B1 (T1 evict 흔적)      +  B2 (T2 evict 흔적)

-> B1 히트 시: T1 공간 확대 (recency 우선)
-> B2 히트 시: T2 공간 확대 (frequency 우선)
-> 워크로드에 따라 자동으로 균형 조정
```

#### LRU와 비교

| 항목 | LRU | ARC |
|------|-----|-----|
| 재사용 빈도 고려 | X | O |
| 워크로드 자동 적응 | X | O |
| 구현 복잡도 | 낮음 | 중간 |
| 메타데이터 오버헤드 | 낮음 | 중간 (Ghost 리스트) |
| Scan 오염 저항성 | 약함 | 중간 |

#### 언제 유리한가?

- 시스템 프롬프트처럼 **자주 반복 접근되는 블록**이 많을 때
- 요청 유형이 섞여 있어 패턴을 예측하기 어려울 때

---

### 전략 2: CLOCK (Second Chance) — 더 가벼운 LRU 근사

#### 핵심 아이디어

LRU의 단점 중 하나는 블록에 접근할 때마다 **연결 리스트를 업데이트**해야 한다는 것입니다.
CLOCK은 이 비용을 줄이면서 LRU와 비슷한 성능을 내는 알고리즘입니다.

#### 작동 방식

각 블록에 **참조 비트(reference bit)** 를 하나 붙입니다.

```
블록 배열 (원형으로 배치)
[블록A: 1] -> [블록B: 0] -> [블록C: 1] -> [블록D: 0] -> ...
                ^
               시계 바늘(pointer)
```

- 블록에 접근하면: 참조 비트를 **1로 설정** (연결 리스트 조작 불필요)
- 블록을 버려야 할 때: 시계 바늘을 돌리면서 탐색
  - 비트 = 1 → **0으로 초기화** 후 패스 (한 번 더 기회 줌)
  - 비트 = 0 → **이 블록을 evict** (최근에 아무도 안 씀)

#### LRU와 비교

| 항목 | LRU | CLOCK |
|------|-----|-------|
| 접근 시 비용 | O(1) 연결 리스트 조작 | O(1) 비트 설정만 (더 가벼움) |
| Eviction 시 비용 | O(1) | O(n) 최악의 경우 탐색 |
| 정확도 | 높음 | LRU 근사 (약간 낮음) |
| 구현 복잡도 | 낮음 | 낮음 |
| 캐시 라인 친화성 | 낮음 | 높음 (배열 기반) |

#### 언제 유리한가?

- 요청 처리량이 매우 높아서 **접근 시 업데이트 비용을 줄여야 할 때**
- 정밀도보다 처리량(throughput)이 더 중요한 환경

---

### 전략 3: S3-FIFO — 스캔 오염에 강한 현대적 알고리즘

#### 핵심 아이디어

2023년 발표된 논문 "FIFO Queues are All You Need for Cache Eviction"에서 제안된 알고리즘입니다.
LRU가 취약한 **스캔 오염(scan pollution)** 문제를 해결합니다.

**스캔 오염이란?** 일회성으로 긴 문서를 처리하는 요청이 들어오면,
그 요청의 블록들이 캐시를 가득 채워 다른 유용한 블록들을 밀어내는 현상입니다.

#### 3개의 큐 구조

```
[Small 큐 (10%)] -> [Main 큐 (90%)] -> [Ghost 큐 (흔적만)]

신규 블록 → Small 큐 (FIFO)
  └─ Small에서 또 접근됨 → Main 큐로 승격
  └─ Small에서 evict → Ghost에 흔적 저장

Main 큐 내 eviction:
  └─ 접근 기록 있음 → 기록 초기화 후 Main 꼬리로 재삽입
  └─ 접근 기록 없음 → evict
```

#### LRU와 비교

| 항목 | LRU | S3-FIFO |
|------|-----|---------|
| 스캔 오염 저항 | 없음 | 강함 (Small 큐가 필터 역할) |
| 인기 블록 보호 | 약함 | 강함 (Main 큐에서 보호) |
| 구현 복잡도 | 낮음 | 낮음 (FIFO 큐 3개) |
| 병렬처리 친화성 | 낮음 | 높음 (큐별 독립 락 가능) |
| 메모리 오버헤드 | 낮음 | 중간 (Ghost 큐) |

#### 언제 유리한가?

- 긴 문서 처리, RAG(검색 증강 생성) 등 **일회성 긴 요청이 많은 환경**
- 시스템 프롬프트나 인기 캐릭터 프롬프트 등 **반복 접근 블록과 일회성 블록이 섞인 환경**

---

### 전략 4: Prefix-Depth-Aware Eviction (LLM 특화 전략)

#### 핵심 아이디어

vLLM은 이미 "꼬리 블록을 먼저 버린다"는 규칙을 갖고 있습니다.
이를 더 발전시켜 **"얼마나 많은 요청이 이 블록을 공유하고 있는가"** 를 eviction 기준에 반영하는 전략입니다.

#### 작동 방식

```
[시스템 프롬프트 블록 0] <- 모든 요청이 공유 (공유 카운트 = 100)
[시스템 프롬프트 블록 1] <- 모든 요청이 공유 (공유 카운트 = 100)
[사용자 A의 대화 블록  ] <- 1개 요청만 사용  (공유 카운트 = 1)
[사용자 B의 대화 블록  ] <- 1개 요청만 사용  (공유 카운트 = 1)
```

공유 카운트가 낮은 블록(= 혼자만 쓰는 블록)을 먼저 버립니다.
같은 공유 카운트라면 LRU 순서를 따릅니다.

eviction 우선순위 = `1 / (공유 카운트 * 최근성 점수)`

#### LRU와 비교

| 항목 | LRU | Prefix-Depth-Aware |
|------|-----|-----|
| Prefix hit rate | 중간 | 높음 |
| 공유 블록 보호 | 약함 | 강함 |
| 구현 복잡도 | 낮음 | 중간 (공유 카운트 집계 필요) |
| 시스템 프롬프트 유지 | 보장 안 됨 | 보장됨 |

#### 언제 유리한가?

- **동일한 시스템 프롬프트**를 사용하는 요청이 많을 때 (챗봇, 공통 RAG 컨텍스트)
- **공통 prefix 비율이 높은** 서비스 환경

---

### 전략 5: W-TinyLFU (Window TinyLFU) — 높은 적중률의 빈도 기반 전략

#### 핵심 아이디어

Java의 Caffeine 캐시 라이브러리에서 사용되어 검증된 알고리즘입니다.
**접근 빈도(frequency)** 를 핵심 기준으로 삼되, 메모리를 거의 쓰지 않는 방법으로 측정합니다.

#### 구조

```
Window LRU (전체의 1%)
  └─ 새 블록은 여기 먼저 들어옴
  └─ 여기서 evict된 블록은 Probation과 경쟁

Protected LRU (전체의 80%)
  └─ Probation에서 재접근되면 여기로 승격
  └─ 고빈도 블록들의 안전지대

Probation LRU (전체의 19%)
  └─ Window에서 내려온 블록 대기소
  └─ 재접근 없이 evict되면 탈락
```

빈도 측정은 **Count-Min Sketch** 라는 확률적 자료구조로 합니다.
정확한 카운터를 블록마다 달지 않아도 메모리 효율적으로 빈도를 추정할 수 있습니다.

```
Count-Min Sketch: 적은 메모리로 접근 빈도를 추정하는 자료구조
(정확하진 않지만 실용적으로 충분히 정확함)

블록 A: 해시 → 카운터 배열에서 빈도 추정
블록 B: 해시 → 카운터 배열에서 빈도 추정
```

#### LRU와 비교

| 항목 | LRU | W-TinyLFU |
|------|-----|-----------|
| 빈도 반영 | X | O (Count-Min Sketch) |
| Scan 오염 저항 | 없음 | 강함 (Window가 필터) |
| 인기 블록 보호 | 약함 | 매우 강함 (Protected 구역) |
| 구현 복잡도 | 낮음 | 높음 |
| 메모리 오버헤드 | 낮음 | 중간 (Sketch + 3개 큐) |
| 실사용 적중률 | 기준 | 대부분의 워크로드에서 LRU보다 높음 |

#### 언제 유리한가?

- **다양한 요청 패턴**이 혼재하고 최고의 적중률을 원할 때
- 특정 블록이 매우 자주 반복 접근되는 환경

---

## 전략별 종합 비교

| 전략 | Scan 저항 | 빈도 반영 | Prefix 보호 | 구현 난이도 | 추천 상황 |
|------|-----------|-----------|-------------|-------------|-----------|
| **LRU** (현재) | 없음 | X | 부분적 | 쉬움 | 일반적인 기본값 |
| **ARC** | 중간 | O | 중간 | 중간 | 패턴 혼재 환경 |
| **CLOCK** | 없음 | X | 부분적 | 쉬움 | 처리량 중시 환경 |
| **S3-FIFO** | 강함 | 부분 | 중간 | 쉬움 | 일회성 긴 요청 많을 때 |
| **Prefix-Depth-Aware** | 없음 | X | 매우 강함 | 중간 | 공통 prefix 비율 높을 때 |
| **W-TinyLFU** | 강함 | O | 중간 | 어려움 | 범용 고성능 환경 |

---

## 어떤 전략을 골라야 할까?

```
내 서비스에 맞는 전략 선택 가이드:

Q1. 모든 요청이 같은 시스템 프롬프트를 쓰나?
  YES → Prefix-Depth-Aware 또는 ARC 고려

Q2. 긴 문서 처리(RAG)나 일회성 긴 요청이 많나?
  YES → S3-FIFO 또는 W-TinyLFU 고려

Q3. 처리량이 너무 중요해서 캐시 업데이트 오버헤드도 줄여야 하나?
  YES → CLOCK 고려

Q4. 그냥 전반적으로 적중률을 높이고 싶나?
  → W-TinyLFU (단, 구현 복잡도 감수)

Q5. 위 상황 다 해당 없고 단순하게 가고 싶나?
  → 현재의 LRU 유지
```

---

## 핵심 파일 경로

| 파일 | 역할 |
|------|------|
| `vllm/v1/core/kv_cache_utils.py` | `FreeKVCacheBlockQueue` (GPU LRU 구현) |
| `vllm/v1/core/block_pool.py` | `BlockPool` (GPU 블록 할당/해제/eviction 관리) |
| `vllm/v1/kv_offload/lru_manager.py` | `LRUOffloadingManager` (오프로딩 LRU) |
| `vllm/v1/kv_offload/arc_manager.py` | `ARCOffloadingManager` (오프로딩 ARC) |

---

# 페이지 교체 외에 추론 속도를 높이는 정책 변경 방안

페이지 교체 알고리즘은 KV 캐시 관리의 일부에 불과합니다.
vLLM에는 그 외에도 **스케줄링, 배치 처리, 투기적 디코딩** 등 추론 속도에 큰 영향을 주는 정책들이 있습니다.

---

## 영역 1: 스케줄링 정책 (Scheduling Policy)

### 현재 상태

vLLM은 두 가지 스케줄링 정책을 지원합니다 ([request_queue.py](vllm/v1/core/sched/request_queue.py)):

| 정책 | 설명 |
|------|------|
| **FCFS** (기본값) | 먼저 온 요청을 먼저 처리 |
| **Priority** | 우선순위 숫자가 낮은 요청을 먼저 처리 |

### 개선 가능한 방향

#### 방향 1-A: Prefix-Cache-Aware 스케줄링

```
현재 FCFS: [요청A(prefix 미스)] -> [요청B(prefix 히트)] -> [요청C(prefix 히트)]
                                    처리 순서: A -> B -> C
                                    낭비: A가 캐시를 덮어써서 B, C의 히트율 하락 가능

개선된 스케줄링: prefix 히트가 많은 요청을 우선 처리
  -> B, C를 먼저 처리해서 캐시 히트 최대화
  -> A는 나중에 처리 (어차피 재계산 필요)
```

- **구현 위치**: `SchedulingPolicy` Enum에 새 정책 추가, `FCFSRequestQueue` 대신 prefix hit 수 기반 정렬 큐 사용
- **장점**: GPU 재계산(prefill) 시간 감소, 전체 처리량 향상
- **단점**: FCFS의 공정성(fairness) 손실, 구현 복잡도 증가

#### 방향 1-B: SJF (Shortest Job First) / 길이 예측 기반

```
아이디어: 짧은 응답을 내는 요청을 먼저 처리
  -> 평균 대기 시간(latency) 감소
  -> GPU 메모리 점유 시간 단축 -> 더 많은 요청 병렬 처리 가능

문제: 응답 길이를 미리 알 수 없음
  -> max_tokens 파라미터나 요청 타입으로 추정
```

- **관련 설정**: `policy: SchedulerPolicy` ([scheduler.py:103](vllm/config/scheduler.py#L103))

---

## 영역 2: Chunked Prefill 파라미터 조정

### Chunked Prefill이란?

긴 프롬프트를 처리할 때, 한 번에 다 처리하는 게 아니라 **여러 조각(chunk)으로 나눠서** 처리하는 방식입니다.

```
일반 방식:
  Step 1: [긴 프롬프트 전체 처리... GPU 독점] -> 다른 요청 대기
  Step 2: [디코딩]

Chunked Prefill:
  Step 1: [프롬프트 앞 부분] + [다른 요청 디코딩] -> 동시 처리
  Step 2: [프롬프트 뒷 부분] + [다른 요청 디코딩]
  Step 3: [디코딩]
```

대기 중인 요청들의 **응답 지연(TTFT: Time To First Token)** 을 줄여줍니다.

### 핵심 파라미터 ([scheduler.py](vllm/config/scheduler.py))

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `enable_chunked_prefill` | `True` | Chunked Prefill 활성화 여부 |
| `max_num_batched_tokens` | `2048` | 한 스텝에서 처리할 최대 토큰 수 |
| `max_num_seqs` | `128` | 동시에 처리할 최대 요청 수 |
| `long_prefill_token_threshold` | `max_model_len * 0.04` | 이 값 이상이면 "긴 요청"으로 분류 |
| `max_long_partial_prefills` | `1` | 동시에 처리할 긴 요청의 최대 수 |
| `max_num_partial_prefills` | `1` | 동시에 prefill 중인 요청의 최대 수 |

### 튜닝 전략

```
처리량(throughput) 최대화:
  max_num_batched_tokens ↑ (크게)
  -> 한 번에 더 많은 토큰 처리
  -> GPU 활용률 향상
  -> 단, 첫 응답 지연(TTFT) 증가

응답 지연(latency) 최소화:
  max_num_batched_tokens ↓ (작게)
  max_long_partial_prefills = 1 (긴 요청이 짧은 요청을 막지 않도록)
  long_prefill_token_threshold ↓ (더 많은 요청을 "긴 요청"으로 분류)
  -> 짧은 요청이 긴 요청보다 먼저 처리될 기회 증가
```

---

## 영역 3: Async Scheduling (비동기 스케줄링)

### 현재 상태

기본 스케줄러는 **GPU 계산이 끝난 후** 다음 배치를 스케줄합니다.

```
동기 방식:
  [GPU 계산] -> [스케줄링] -> [GPU 계산] -> [스케줄링] ...
                ^^ 이 시간 동안 GPU가 놀고 있음 (idle gap)
```

### Async Scheduling의 개선

`AsyncScheduler`([async_scheduler.py](vllm/v1/core/sched/async_scheduler.py))는 GPU 계산과 스케줄링을 **겹쳐서** 실행합니다.

```
비동기 방식:
  [GPU 계산 N] + [스케줄링 N+1] -> [GPU 계산 N+1] + [스케줄링 N+2] ...
  ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
  동시에 실행!  GPU idle 없음
```

- **설정**: `async_scheduling: bool` ([scheduler.py:133](vllm/config/scheduler.py#L133))
- **장점**: GPU idle 시간 제거 → 처리량 향상
- **주의**: speculative decoding, pipeline parallelism과 함께 사용 시 자동 비활성화

---

## 영역 4: Speculative Decoding (투기적 디코딩)

### 핵심 아이디어

LLM은 한 번에 토큰 하나씩 생성하는데, 이게 매우 느립니다.
투기적 디코딩은 **작은 모델이 여러 토큰을 먼저 예측**하고,
**큰 모델이 한 번에 검증**하는 방식으로 속도를 높입니다.

```
일반 방식:
  큰 모델 -> 토큰1 -> 큰 모델 -> 토큰2 -> 큰 모델 -> 토큰3
  (매번 큰 모델 실행)

투기적 디코딩:
  작은 모델 -> [토큰1, 토큰2, 토큰3, 토큰4, 토큰5] 예측
  큰 모델  -> 한 번에 5개 검증
            -> "토큰1 ✓, 토큰2 ✓, 토큰3 ✓, 토큰4 ✗" → 토큰1~3 채택, 토큰4부터 재생성
  net 결과: 큰 모델 1회 실행으로 최대 3개 토큰 생성
```

### vLLM에서 지원하는 방식 ([vllm/v1/spec_decode/](vllm/v1/spec_decode/))

| 방식 | 파일 | 설명 |
|------|------|------|
| **EAGLE / EAGLE3** | `eagle.py` | 별도 draft 모델 학습, 높은 수락률 |
| **Medusa** | `medusa.py` | 메인 모델에 head 추가, 병렬 예측 |
| **n-gram** | `ngram_proposer.py` | 프롬프트 내 반복 패턴 활용, 모델 불필요 |
| **Suffix Decoding** | `suffix_decoding.py` | 이전 출력 히스토리에서 suffix 매칭 |

### 성능 효과

```
수락률(acceptance rate)에 따라 속도가 결정됨:
  - 수락률 높음 (0.9) → 최대 N배 속도 향상 (N = 투기 토큰 수)
  - 수락률 낮음 (0.3) → 오히려 느려질 수도 있음

n-gram/suffix decoding:
  - 별도 모델 없이 적용 가능 → 시도 비용이 낮음
  - 코드 생성, 반복 패턴이 많은 텍스트에서 효과적
```

- **관련 설정**: `num_speculative_tokens`, `speculative_model`

---

## 영역 5: KV 캐시 블록 크기 (Block Size)

### 현재 상태

vLLM은 KV 캐시를 고정 크기 블록으로 나눠서 관리합니다.
블록 크기는 [cache.py](vllm/config/cache.py)의 `block_size`로 설정합니다.

### 블록 크기의 영향

```
작은 블록 크기 (예: 8 tokens):
  장점: 내부 단편화(internal fragmentation) 감소
       (블록 내 빈 공간이 적음 → 메모리 효율 높음)
       Prefix 캐시 히트 세밀도 증가
  단점: 블록 수 증가 → 관리 오버헤드 증가

큰 블록 크기 (예: 32 tokens):
  장점: 블록 관리 오버헤드 감소
       연속 메모리 접근 → GPU 캐시 효율 향상
  단점: 내부 단편화 증가
       (예: 10 토큰 요청이 32 토큰 블록 점유)
```

| 상황 | 권장 블록 크기 |
|------|--------------|
| 짧은 요청이 많음 | 작게 (8~16) |
| 긴 요청이 많음 | 크게 (32~64) |
| Prefix 캐시 히트율 중시 | 작게 |
| GPU 메모리 대역폭 효율 중시 | 크게 |

---

## 영역 6: Prefix Caching 해시 알고리즘

### 현재 상태

Prefix Caching은 블록의 내용을 해시값으로 식별합니다.
vLLM은 두 가지 해시 알고리즘을 지원합니다 ([cache.py:79](vllm/config/cache.py#L79)):

| 알고리즘 | 특징 |
|----------|------|
| `sha256` (기본값) | 느리지만 충돌 없음, 재현 가능 |
| `xxhash` | 빠름, 암호학적 안전성은 낮음 |

### 성능 영향

```
캐시 히트 체크는 매 요청마다 수행됨
  -> 해시 계산이 빠를수록 스케줄링 오버헤드 감소

대규모 서빙 환경에서 xxhash가 sha256보다 훨씬 빠름
단, 여러 프로세스 간 해시값 공유가 필요할 때는 PYTHONHASHSEED 설정 필요
```

- **설정**: `prefix_caching_hash_algo: PrefixCachingHashAlgo`

---

## 영역 7: Preemption(선점) 정책

### 현재 상태

GPU 메모리가 부족할 때, 실행 중인 요청을 **일시 중단(preempt)** 해야 합니다.
현재 vLLM의 preemption은 스케줄링 정책에 따라 다릅니다 ([scheduler.py:366-395](vllm/v1/core/sched/scheduler.py#L366-L395)):

```python
# FCFS 정책: 마지막(가장 최근) 요청을 preempt
preempted_req = self.running.pop()  # 리스트 맨 뒤

# Priority 정책: 가장 낮은 우선순위 요청을 preempt
preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
```

### 개선 가능한 방향

#### Completion-Rate-Aware Preemption

```
현재 방식: 마지막에 들어온 요청을 preempt
           → 거의 완성된 요청이 중단될 수 있음 (낭비)

개선 방식: 완료율이 가장 낮은 요청을 preempt
  완료율 = num_computed_tokens / num_total_tokens
  → 이미 많이 진행된 요청을 살리고, 초기 요청을 중단
  → 전체 KV 캐시 낭비 감소
```

---

## 영역 8: KV Cache Offloading 활용

### 핵심 아이디어

GPU 메모리가 부족할 때 KV 블록을 버리는 대신, **CPU 메모리로 오프로딩**합니다.
나중에 그 요청이 다시 필요해지면 GPU로 다시 올립니다.

```
KV 오프로딩 없음:
  GPU 꽉 참 → 요청 preempt → KV 캐시 삭제 → 나중에 재계산 필요

KV 오프로딩 있음:
  GPU 꽉 참 → KV를 CPU로 이동 → GPU 공간 확보
  나중에 요청 재개 → CPU에서 GPU로 로딩 (재계산보다 빠름)
```

- **관련 파일**: `vllm/v1/kv_offload/` (LRU, ARC Manager)
- **설정**: `swap_space` ([cache.py:58](vllm/config/cache.py#L58)) - CPU 스왑 공간 크기(GiB)
- **장점**: 재계산 비용 절감, 더 많은 요청 동시 처리
- **단점**: CPU-GPU 메모리 전송 지연 발생

---

## 영역 9: Streaming 응답 간격 (stream_interval)

### 현재 상태

클라이언트에 토큰을 스트리밍할 때의 버퍼 크기입니다 ([scheduler.py:141](vllm/config/scheduler.py#L141)).

```
stream_interval = 1 (기본값):
  토큰 생성 즉시 전송 → 부드러운 스트리밍
  단, 매 토큰마다 네트워크/시스템 오버헤드 발생

stream_interval = 10:
  10개 토큰을 모아서 전송 → 오버헤드 감소
  단, 사용자가 느끼는 스트리밍이 덜 부드러움
```

처리량(throughput) 중시 환경에서는 `stream_interval`을 올리면
**CPU 오버헤드가 줄어** 전체 처리량이 향상됩니다.

---

## 전체 정책 변경 종합 가이드

```
목표: 처리량(Throughput) 최대화
  ✓ max_num_batched_tokens ↑ (크게)
  ✓ async_scheduling = True
  ✓ enable_chunked_prefill = True
  ✓ stream_interval ↑ (10~20)
  ✓ 투기적 디코딩 활성화 (n-gram부터 시도)
  ✓ prefix_caching_hash_algo = xxhash

목표: 응답 지연(Latency) 최소화
  ✓ max_long_partial_prefills = 1
  ✓ long_prefill_token_threshold ↓ (더 적극적 chunking)
  ✓ async_scheduling = True
  ✓ 투기적 디코딩 (EAGLE) — 고수락률 시

목표: Prefix Cache 히트율 최대화
  ✓ enable_prefix_caching = True (기본)
  ✓ Prefix-Cache-Aware 스케줄링 (직접 구현 필요)
  ✓ GPU 캐시 교체 알고리즘을 Prefix-Depth-Aware로 변경
  ✓ block_size 최적화 (공통 prefix 길이에 맞춤)

목표: 메모리 효율 최대화
  ✓ KV Offloading (swap_space 충분히 할당)
  ✓ block_size 조정 (요청 길이 분포에 맞춤)
  ✓ ARC로 GPU 캐시 교체 전략 변경
```

### 관련 핵심 파일 경로

| 파일 | 관련 정책 |
|------|-----------|
| `vllm/config/scheduler.py` | 스케줄링 정책, chunked prefill, async scheduling |
| `vllm/config/cache.py` | 블록 크기, prefix caching, swap space |
| `vllm/v1/core/sched/request_queue.py` | FCFS / Priority 큐 구현 |
| `vllm/v1/core/sched/scheduler.py` | 메인 스케줄링 로직, preemption 정책 |
| `vllm/v1/core/sched/async_scheduler.py` | 비동기 스케줄링 |
| `vllm/v1/spec_decode/` | 투기적 디코딩 (EAGLE, Medusa, n-gram, suffix) |
| `vllm/v1/kv_offload/` | KV Cache CPU 오프로딩 |
