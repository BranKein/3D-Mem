# RefON 지시문은 로봇이 움직이기 전에 풀 수 있는가?

*English version: [`refon_feasibility_analysis.md`](refon_feasibility_analysis.md)*

RefON 지시문은 조응적(anaphoric)입니다: *"Find A1."*, *"Find the 2nd one again."*,
*"Go back to the previous one."* 이런 지시문에서 네비게이션이 실패했을 때 원인은 둘 중
하나인데, 네비게이션 점수는 그 둘을 구분해주지 못합니다.

1. 에이전트가 지시문이 **어떤 물체**를 뜻하는지 애초에 알아내지 못했거나,
2. 알고도 **찾지** 못했거나.

아래의 모든 것은 (1)만 떼어냅니다. habitat 없음, 이미지 없음, 네비게이션 없음 — 에이전트는
지시문만 봅니다. 그래서 결과는 **상한선**입니다: 여기서 실패한 subgoal은 지각이나 탐색과
관련된 어떤 이유로도 네비게이션에서 성공할 수 없습니다.

두 개의 probe를 씁니다. 오픈웨이트 모델은 RTX 4090 한 장 위의 `ollama`로,
`claude-haiku-4-5`는 동일한 프롬프트와 채점기로 Anthropic API를 통해 돌립니다.

| probe | 질문 | 답의 형태 | 실행 파일 |
|---|---|---|---|
| **referent** | 이 지시문은 어떤 물체를 뜻하는가? | `new` / `back_reference` + 지시문 번호 / `no_object`, 그리고 물체의 카테고리 | `run_refonbench_feasibility.py` |
| **destination** | 로봇은 어디로 가는가? | `explore` / `(x, y)` / `infeasible` | `run_refonbench_feasibility_nav.py` |

프롬프트 전문과 성공 응답 예시는 §10에 있습니다.

---

## 1. 방법

### 1.1 referent probe

지시문은 **누적 제시(incremental)** 됩니다 — subgoal *i* 시점에 모델은 지시문 1..*i*를 보고
*i*에 대해 질문받습니다. 실제 네비게이션 실행이 그 시점에 가진 정보와 정확히 같습니다.
(에피소드 전체를 한 번에 보여주는 `all_at_once` 모드도 있지만 여기 비교에는 포함하지 않았습니다
— 초기 수치가 왜 못 쓸 것이었는지는 §8의 첫 항목을 보십시오.)

채점은 두 개의 부분 점수로 나뉘며 따로 보고합니다. 모델이 올바른 물체를 짚고도 이름을 못 댈 수
있기 때문입니다.

- **referent SR** — 올바른 물체를 짚었는가?
- **category SR** — 그 물체의 카테고리를 제대로 말했는가?
- **joint SR** — 둘 다.

두 가지 규칙이 중요합니다.

**같은 `object_id`에 도달하는 *이전* 지시문이면 무엇이든 정답으로 인정합니다.** 한 에피소드가
같은 물체를 여러 번 방문할 수 있어서 "2번째 것"과 "3번째 것"이 동시에 같은 referent에 대해
참일 수 있습니다. 다만 subgoal **자기 자신의 번호는 의도적으로 제외**합니다 — 자기 번호는
당연히 올바른 `object_id`를 갖고 있으므로, 이를 인정하면 "이건 새 물체다"라는 답이 back
reference의 올바른 해소로 채점되어 버립니다.

**goal-absent 세 종류는 텍스트-온리 probe에게 같은 질문이 아닙니다.**

| 종류 | 예시 | 기대 답 | 이유 |
|---|---|---|---|
| `GA_unbound_alias` | "Find Z1." | `no_object` | 별칭이 바인딩된 적 없음 — 텍스트가 그렇게 말함 |
| `GA_invalid_ordinal` | 6번 방문 후 "Find the 8th one again." | `no_object` | 범위 초과 — 텍스트가 그렇게 말함 |
| `GA_absent_object` | "Find the chandelier." | `new` + "chandelier" | **씬**에 샹들리에가 없어서 goal-absent인데, 모델은 그 사실을 들은 적이 없음 |

세 번째에 `no_object`를 요구하는 것은 모델이 갖고 있지 않은 사실을 채점하는 것입니다.

### 1.2 destination probe

에피소드의 모든 물체에 가상의 `(x, y)`를 부여합니다. 에피소드마다 시드를 고정해 모든 모델이
같은 지도를 봅니다. 위치는 **이전 subgoal들이 그곳에 도달하면서만** 공개되므로, 매 스텝에서
정확히 하나의 행동만이 정답입니다: 아무도 방문하지 않은 물체에는 `explore`, 이미 찾은 물체에는
로그에서 복사한 좌표, 수행 불가능한 지시문에는 `infeasible`. 좌표는 서로 1.0 이상 떨어뜨려
근소한 오차가 정답으로 채점되지 않게 했습니다.

`GA_absent_object`는 여기서 **두 턴**을 받습니다. 그것이 이를 공정하게 채점하는 유일한
방법입니다.

```
턴 1     "Find the chandelier."                     -> explore      (정답)
환경     로봇이 탐색했고 보고: not_found
턴 2     같은 지시문 + 그 결과                        -> infeasible   (정답)
```

두 번째 턴은 `GA_absent_object/not_found`라는 별도 스타일로 보고합니다.

로그는 **모델이 뭐라고 답했는지가 아니라 실제로 일어난 일**을 담습니다. 따라서 모든 턴이 독립
질의이고, 실행이 병렬화되며, subgoal 3의 오답 하나가 4..N에서 아무것도 측정하지 못하는 연쇄
실패로 번지지 않습니다.

### 1.3 데이터셋

둘 다 무료 HM3D 예제 씬 `00861-GLAQ4DNUx5U` 위에 만들었고, 스타일별 SR의 표본 수가 비슷해지도록
instruction style을 균등화했습니다(생성기의 `even_styles` 차원).

| 세트 | 에피소드 | subgoal | 길이 | 참조 거리 | 설정 파일 |
|---|---|---|---|---|---|
| **balanced** | 200 | 1308 | 4–8 | ≤7 | `generator.example_even.json` |
| **long** | 100 | 2273 | 5/14/23/32/41/50 | ≤49 | `generator.example_long.json` |

스타일 비중은 균등 목표 14.3% 대비 balanced에서 12.7–17.2%, long에서 11.7–16.8%에
안착했습니다. 남는 `AB_post` 초과분은 구조적입니다 — `AB_pre` 하나가 바인딩한 별칭을 여러
`AB_post`가 참조할 수 있으므로 설계상 후자가 전자보다 많습니다.

### 1.4 모델

qwen 계열 네 모델이 사다리를 이룹니다. 이들은 **non-thinking**으로 돌아가므로 그 비교는 전체가
하나의 조건입니다(이유는 §7.1). 나머지 둘은 사다리 밖의 참조점입니다.

| 모델 | 크기 | 비고 |
|---|---|---|
| `qwen3.5:2b` | 2B | |
| `qwen3.5:4b` | 4B | |
| `qwen2.5vl:7b` | 7B | 이전 세대 |
| `qwen3.5:9b` | 9B | |
| `gemma4:26b-a4b-it-qat` | 26B (활성 4B) | thinking 켜짐 — 직접 비교 불가 |
| `claude-haiku-4-5` | 비공개 | API 모델, thinking 비활성화 |

Anthropic은 어떤 Claude 모델의 파라미터 수도 공개하지 않으므로 `claude-haiku-4-5`는 크기 축에
놓을 수 없고 §5.2의 스케일 사다리 주장에서 제외됩니다. 여기서는 **상단 참조점**으로 있습니다:
참조 해소 단계가 크기 제약을 받지 않을 때 같은 프롬프트가 몇 점을 받는가.

`qwen3.5:0.8b`는 제외했습니다. 토큰 예산 전체를 thinking에 쓰고 빈 content를 반환합니다
(완료 토큰 16384, `reasoning` 6만 6천 자, 호출당 240초).
`qwen3-vl:8b`는 non-thinking 비교에서 제외했습니다. non-thinking 모드로 만들 수가 없기
때문입니다 — `think=False`, `/no_think`, `enable_thinking=False`, `reasoning_effort=none`이
모두 무시됩니다(플래그 없이 5089 토큰, 플래그를 줘도 5701 토큰).

### 1.5 스타일 보고 규약

`AB_pre`와 `AR_pre`는 전 구간에서 `S`로 접습니다. `scripts/summarize_refonbench.py`가 하는
것과 같습니다. 셋 모두 자기 목표를 스스로 이름으로 부르고 답의 형태도 같으며 — `AB_pre`는
별칭 바인딩("Let's call it A1.")을 덧붙일 뿐이고 그건 해소할 대상이 아닙니다. 그 결과 `S`는
balanced 세트에서 534 표본, long에서 958 표본이 됩니다.

`AB_pre+OR_post`는 분리해 둡니다. 별칭을 바인딩하면서 *동시에* 되짚기 때문에, 이를 접으면
조응이 없어야 할 행에 조응 케이스가 섞입니다. 비교 스크립트에 `--merge-roles`를 주면 이 그룹핑이
재현되고, 빼면 10개 스타일 전부가 보입니다.

---

## 2. referent probe, balanced 세트 (1308 subgoal)

여기서 `S`는 `AB_pre`와 `AR_pre`를 포함합니다(§1.5).

| 스타일 | n | 2b | 4b | 7b | 9b | gemma4 26b† | haiku 4.5‡ |
|---|---|---|---|---|---|---|---|
| **S** | 534 | 79.2 | 90.4 | **99.3** | 95.9 | 91.8 | 91.4 |
| AB_post | 220 | 0.0 | 71.4 | 53.2 | 85.9 | **97.7** | 95.5 |
| AR_post | 185 | 2.7 | 43.8 | 14.1 | 64.3 | **96.8** | 88.1 |
| OR_post | 178 | 0.0 | 52.2 | 11.8 | 84.8 | 94.9 | **96.1** |
| AB_pre+OR_post | 162 | 0.0 | 22.2 | 2.5 | 75.3 | **97.5** | 94.4 |
| GA_absent_object | 9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| GA_invalid_ordinal | 13 | 0.0 | 15.4 | 0.0 | 84.6 | **100.0** | **100.0** |
| GA_unbound_alias | 7 | 0.0 | 0.0 | 0.0 | 28.6 | **100.0** | **100.0** |
| **전체** | 1308 | **33.4** | **65.8** | **54.1** | **85.2** | **94.8** | **92.8** |
| *back-reference* | 745 | 0.7 | 49.3 | 22.6 | 78.0 | **96.8** | 93.6 |

† thinking 켜짐. 사다리의 일부가 아니라 참고용.
‡ API 모델, 크기 비공개. §1.4 참고.

**첫 행과 마지막 행의 대비가 전부입니다.** 모든 모델이 자기 목표를 스스로 부르는 지시문에는
쓸 만하고, 그렇지 않은 지시문에서 거의 전적으로 갈립니다: `S`에서는 사다리 전체가 79→96인데
back reference에서는 0.7→78입니다.

두 참조점은 이 패턴을 뒤집습니다. gemma4와 haiku 모두 `S`(91.8, 91.4)에서 back
reference(96.8, 93.6)보다 **낮게** 나옵니다. 퇴보가 아닙니다 — §7.2가 `S` 행에 어떤 능력으로도
넘을 수 없는 100% 미만의 천장이 있는 이유를 설명합니다.

---

## 3. destination probe, balanced 세트 (1317 턴)

| | 2b | 4b | 7b | 9b | haiku 4.5 |
|---|---|---|---|---|---|
| action SR | 28.9 | 85.0 | 71.1 | 90.2 | **95.2** |
| coord SR | 43.1 | 68.2 | 65.9 | 78.8 | **90.7** |
| **joint SR** | **28.3** | **63.8** | **60.7** | **77.9** | **90.6** |
| *S* | 63.9 | 80.5 | 85.0 | 91.0 | **91.9** |
| *back-reference* | 0.4 | 51.1 | 45.5 | 68.1 | **89.1** |

스타일별 action 대 joint:

| 스타일 | n | 9b action | 9b joint | 9b 격차 | haiku action | haiku joint | haiku 격차 |
|---|---|---|---|---|---|---|---|
| S | 534 | 91.0 | 91.0 | 0.0 | 91.9 | 91.9 | 0.0 |
| AB_post | 220 | 90.0 | 78.6 | −11.4 | 100.0 | 100.0 | 0.0 |
| **AR_post** | 185 | 90.3 | 49.7 | **−40.6** | 97.3 | 75.7 | **−21.6** |
| OR_post | 178 | 87.1 | 69.7 | −17.4 | 94.4 | 89.3 | −5.1 |
| AB_pre+OR_post | 162 | 92.0 | 72.8 | −19.2 | 96.9 | 89.5 | −7.4 |

`S`는 구조상 격차가 없습니다 — `explore`가 답의 전부라 틀릴 좌표가 존재하지 않습니다.
back-reference 행은 전부 격차가 있습니다. `AB_post`의 haiku만 예외입니다.

**돌아가기로 결정하는 것이 어디로 돌아갈지 아는 것보다 쉽습니다.** `AR_post`에서 모델은 열 번 중
아홉 번 올바른 *종류*의 행동을 고르고도 그중 절반에서 목적지를 틀립니다. Haiku는 모든 곳에서
격차를 좁히지만 없애지는 못하며, `AR_post`("go back to the previous one", "find the one before
that")는 여전히 압도적으로 그의 최악 스타일입니다 — 다른 모든 스타일이 89–100%인데 75.7%.
가장 마지막에 자리를 잡는 것은 별칭 바인딩이나 서수가 아니라 **상대 조응**입니다.

goal-absent 2턴 프로토콜은 두 능력을 깔끔하게 분리했습니다: `GA_absent_object/not_found`
(탐색 실패 후 포기하기)는 **모든 모델에서 100%**인 반면, `GA_absent_object` 턴 1(결론 내리기
전에 일단 찾아보러 가기)은 **2b와 4b에서 0%**, 9b와 haiku에서 100%입니다. 작은 모델은 곧장
`infeasible`로 건너뜁니다. 한 턴 만에 `infeasible`을 요구했다면 이 사실이 통째로 가려졌을
것입니다.

---

## 4. 긴 에피소드 (2273 subgoal, 길이 5–50)

| 모델 | balanced | long | 하락 |
|---|---|---|---|
| qwen3.5:2b | 33.4 | 15.0 | −18.4 |
| qwen2.5vl:7b | 54.1 | 33.0 | −21.1 |
| qwen3.5:4b | 65.8 | 42.5 | −23.3 |
| qwen3.5:9b | 85.2 | 60.8 | −24.4 |
| claude-haiku-4-5 | 92.8 | **82.5** | **−10.3** |

**오픈웨이트 사다리 안에서는 모든 모델이 18–24점 떨어지고, 가장 큰 모델이 가장 많이
떨어집니다.** 그 사다리 안에서의 스케일은 긴 대화에 대한 강건성을 사주지 않습니다 — 더 높은
출발점을 사줄 뿐입니다.

**Haiku는 이 패턴을 깹니다**: 7.6점 더 높은 출발점에서 10.3점만 잃습니다. 9b가 잃는 것의 절반도
안 됩니다. 즉 이 열화는 **길이에 따른 과제의 본질적 성질이 아니라 그 모델들의 성질**입니다.
크기 축이 포착하지 못하는 무언가가 두 그룹을 가르고 있고, 이는 §5.2의 사다리 주장을 qwen
계열로 한정해야 하는 또 하나의 이유입니다.

스타일별:

| 스타일 | n | 2b | 4b | 7b | 9b | haiku 4.5 |
|---|---|---|---|---|---|---|
| **S** | 958 | 34.6 | 66.4 | 44.2 | 71.0 | **85.1** |
| AB_post | 379 | 0.0 | 44.9 | 45.4 | 70.4 | **89.7** |
| AR_post | 325 | 1.8 | 21.2 | 25.2 | 48.3 | **61.8** |
| OR_post | 332 | 0.0 | 18.7 | 12.0 | 50.0 | **84.0** |
| AB_pre+OR_post | 264 | 0.0 | 6.4 | 10.6 | 36.7 | **86.0** |
| **전체** | 2273 | **15.0** | **42.5** | **33.0** | **60.8** | **82.5** |
| *back-reference* | 1300 | 0.5 | 24.5 | 24.8 | 52.8 | **80.5** |

`S`도 함께 떨어진다는 점에 주목하십시오 — 9b는 95.9 → 71.0, haiku는 91.4 → 85.1. 직접
지시문은 필요한 것을 전부 담고 있는데도 그렇습니다. 길이는 조응 케이스뿐 아니라 쉬운 케이스도,
모든 모델에서 해칩니다.

referent가 얼마나 뒤에 있는지에 대한 joint SR:

| 거리 | 1 | 2 | 3–4 | 5–8 | 9–16 | 17+ |
|---|---|---|---|---|---|---|
| n | 208 | 471 | 152 | 173 | 156 | 140 |
| claude-haiku-4-5 | 92 | 71 | 86 | 82 | 83 | 87 |
| qwen3.5:9b | 75 | 53 | 55 | 44 | 40 | 41 |
| qwen3.5:4b | 34 | 25 | 24 | 25 | 15 | 17 |
| qwen2.5vl:7b | 22 | 27 | 21 | 26 | 24 | 23 |
| qwen3.5:2b | 0 | 1 | 0 | 0 | 0 | 0 |

9b는 참조가 뒤로 갈수록 75%에서 약 40%로 반토막 난 뒤 평평해집니다. 4b와 7b는 거의 평평하고
낮습니다 — 이들은 거리 *때문에* 실패하는 게 아니라 back reference 자체에 실패하고 있습니다.

**Haiku는 평평하고 높습니다**: 거리 1에서 92%인데 거리 17+에서 87%입니다. 지시문 17개를 되짚는
것이 사실상 비용이 되지 않는다는 뜻이고, 그렇다면 위의 긴 에피소드 하락은 **탐색 범위 문제가
아닙니다**. 유일한 함몰은 거리 2 구간(71%)인데, 이 구간은 `AR_post` — "the one before that" —
이 지배하며, 다른 두 실험에서도 haiku의 최악 스타일이었던 바로 그것입니다. 그가 잃는 것은 *먼*
참조가 아니라 *상대적* 참조이고, 상대 참조는 마침 근거리에서 발생합니다.

---

## 5. 교차 발견

### 5.1 세대가 크기를 이긴다

`qwen3.5:4b`는 **파라미터 절반**으로 `qwen2.5vl:7b`를 두 데이터셋 모두에서 이깁니다 —
balanced 65.8 대 54.1, long 42.5 대 33.0. 그리고 `qwen2.5vl:7b`는 자기 목표를 스스로 부르는
지시문에서 **이 연구 최고 모델**입니다(99.3%). 격차는 전적으로 back reference에 몰려 있고
(22.6% 대 49.3%), 그것이 바로 이 벤치마크가 측정하려고 존재하는 능력입니다.

### 5.2 스케일 사다리는 실재하지만 스타일 한정이다

한 계열, 한 조건 안에서: **33.4 → 65.8 → 85.2%** (2b → 4b → 9b). 이득의 거의 전부가 back
reference에 있고(0.7 → 49.3 → 78.0), `S`는 79.2%에서 시작해 95.9%로 기어갑니다. 스케일이
사주는 것은 참조 해소이지 지시문 이해가 아닙니다.

이 주장은 qwen 계열로 한정됩니다. `gemma4:26b-a4b`와 `claude-haiku-4-5`는 이 사다리 위의 점이
아니며(전자는 thinking 켜짐, 후자는 크기 비공개), §4는 사다리 자체의 경향(클수록 길이에 더 많이
열화)이 그 밖으로 확장되지 않음을 보입니다.

### 5.3 두 probe는 난이도 사다리가 아니라 상호보완이다

동일한 1308개 항목을 subgoal 단위로 짝지어 보면:

| 모델 | referent | nav | 둘 다 성공 | probe만 | **nav만** | 둘 다 실패 |
|---|---|---|---|---|---|---|
| qwen3.5:2b | 33.4 | 27.8 | 21.5 | 11.9 | 6.3 | 60.2 |
| qwen2.5vl:7b | 54.1 | **61.2** | 41.0 | 13.1 | **20.2** | 25.8 |
| qwen3.5:4b | 65.8 | 63.5 | 49.2 | 16.6 | 14.3 | 19.9 |
| qwen3.5:9b | 85.2 | 77.8 | 71.4 | 13.8 | 6.3 | 8.4 |
| claude-haiku-4-5 | 92.8 | 90.5 | 87.8 | 5.0 | **2.8** | 4.4 |

(이 표의 nav 열은 §3보다 0.1점 낮게 나올 수 있습니다. 짝짓기가 공유되는 1308개 항목으로
제한되어 `GA_absent_object/not_found` 두 번째 턴이 빠지기 때문입니다.)

destination probe가 referent probe에 한 단계를 더한 것이라면 "nav만"은 0에 가까워야 합니다.
사다리 전체에서 6–20%입니다.

Haiku만이 거의 그렇습니다(2.8%). 그리고 그것이 진짜 난이도 사다리가 가질 모양입니다. 따라서 두
probe의 불일치는 대체로 probe의 성질이 아니라 **능력 아티팩트**입니다: 모델이 좋아질수록 두
질문은 같은 답으로 수렴합니다. 다만 사라지지는 않습니다 — "probe만" 5.0%에 "nav만" 2.8%를
더하면, 90%대에서도 여전히 7.8%의 subgoal이 둘을 갈라놓습니다.

`qwen2.5vl:7b`는 오히려 더 어려워 보이는 probe에서 **더 높은** 점수를 받고(61.2 대 54.1),
스타일별 수치가 그 이유를 말해줍니다. `S`에서는 *잘못된* 방향으로 움직이고(99.3 → 85.0), 모든
back reference에서는 크게 옳은 방향으로 움직입니다: `AB_pre+OR_post` 2.5 → 48.1, `OR_post`
11.8 → 49.4, `AR_post` 14.1 → 40.0. 이 모델은 *"referent는 지시문 3이다"*를 산출하지는 못해도
그 지시문의 좌표를 복사할 수는 있습니다. 낮은 referent 점수는 상당 부분 **형식** 제약이었고,
destination probe가 그것을 독립적으로 입증합니다.

반대 방향으로, `qwen3.5:9b`는 destination probe로 가면서 7.4점을 잃는데 `AR_post`와
`OR_post`에 몰려 있습니다 — referent를 아는 것과 그 좌표를 옮기는 것은 분리 가능한 기술입니다.

**어느 probe도 단독으로는 참조 해소를 깨끗하게 측정하지 못합니다.** 하나는 답 형식에, 다른
하나는 좌표 처리에 오염되어 있습니다.

---

## 6. 한 단계인가, 두 단계인가?

위의 모든 것은 참조 해소를 *고립시켜* 측정합니다. 그것이 시스템을 어떻게 만들어야 하는가에
영향을 줄 때에만 의미가 있는데, 실제로 영향을 줍니다. 현재 RefON 평가는 goal 추론과 네비게이션을
**같은 프롬프트 안에서** 수행하는 VLM **하나**를 돌립니다. 대안은 이를 쪼개는 것입니다:
텍스트만으로 참조를 해소하고, 그 결과 goal을 3D-Mem에 넘기고, 3D-Mem은 평범한 object-goal
navigation을 하게 하는 것.

```
1-step   지시문 ────────────────────────────► VLM (해소 + 네비) ──► 성공?
2-step   지시문 ──► 해소기 (텍스트만) ──► "찾아라: <goal>" ──► 3D-Mem (object-nav) ──► 성공?
                        = S1                                        = S2
```

### 6.1 1-step 파이프라인의 실제 점수

`results/exp_eval_refonbench_default_qwen3_vl_30b` 기준(`qwen3-vl:30b`,
`success_distance` 1.0 m). **37개 subtask만 채점되었고 실행은 부분적**이므로 여기의 모든 수치는
대략 ±11점을 달고 있습니다.

| | n | SR |
|---|---|---|
| **전체** | 37 | **13.5%** (SPL 0.100) |
| merged-`S` — 해소할 참조가 없음, 즉 순수 object-nav | 15 | **20.0%** |
| 참조 스타일 | 22 | **9.1%** |

`success_by_snapshot`은 전 구간 0.0%이고 `success_by_distance`만 값이 잡힙니다.

이 두 행이 **임베디드 참조 해소 계수**를 직접 줍니다: 9.1 / 20.0 = **0.455**. 같은 파이프라인이
20% 확률로 푸는 지시문에 참조 표현을 붙이면 9%로 잘립니다. §2가 같은 급 모델들의 텍스트-온리
수치를 **0.93–0.95**로 놓습니다.

### 6.2 3D-Mem이 논문에서 보고한 object-nav 성공률

3D-Mem 논문 Table 3 (GOAT-Bench `val_unseen`, GPT-4o):

| goal 종류 | SR | SPL |
|---|---|---|
| **object category** — S2에 해당하는 것 | **79.2** | 55.8 |
| language | 61.9 | 46.0 |
| image | 65.2 | 44.2 |
| 전체 | 69.1 | 48.9 |

같은 표의 baseline: Explore-EQA 55.0/37.9, ConceptGraph w/ Frontier Snapshots 61.5/45.3,
3D-Mem w/o memory 58.6/38.5. A-EQA(Table 1): LLM-Match 52.6, SPL 42.0.

### 6.3 곱셈

`SR₂ = S1 × S2`이며, S2는 두 가지로 잡습니다. **A** = 논문의 79.2%(다른 모델, 다른 벤치마크 —
낙관적 천장), **B** = 위에서 측정한 merged-`S` 20.0%(같은 모델, 같은 씬, 같은 성공 기준 —
비관적이지만 내부적으로 일관됨).

| 해소기 | S1 | 2-step, anchor A | 2-step, anchor B |
|---|---|---|---|
| gemma4:26b-a4b | 94.8 | 75.1 | 19.0 |
| claude-haiku-4-5 | 92.8 | 73.5 | 18.6 |
| qwen3.5:9b | 85.2 | 67.5 | 17.0 |
| qwen3.5:4b | 65.8 | 52.1 | 13.2 |
| qwen2.5vl:7b | 54.1 | 42.8 | 10.8 |
| qwen3.5:2b | 33.4 | 26.5 | 6.7 |
| *1-step, 실측* | — | *13.5* | *13.5* |

손익분기 S1은 A에서 13.5/79.2 = **17.0%**, B에서 13.5/20.0 = **67.5%**입니다.

### 6.4 왜 이것이 작은 온보드 모델에서 뒤집히지 않는가

이 표를 "2B 해소기는 6.7%니까 작은 하드웨어에서는 1-step이 낫다"로 읽는 것은 오독이고, 그건
표의 잘못입니다. 이 표는 1-step 수치를 30B 측정값에 **고정한 채** 해소기만 바꿉니다. 대신
모델을 고정하십시오:

```
1-step(M) = objnav(M) × r_emb(M)      r_emb  = 지각도 하면서 하는 참조 해소
2-step(M) = objnav(M) × s_text(M)     s_text = 텍스트만으로 하는 참조 해소
```

`objnav(M)`이 약분됩니다. 비교는 **같은 모델의 s_text 대 r_emb**로 환원되며, 이 질문에는 크기
항이 없습니다. 텍스트-온리 해소는 결합 과제의 엄격한 부분문제이므로 `s_text ≥ r_emb`가 모든
스케일에서 성립해야 하고, 측정된 쌍(0.93 대 0.455)은 넓은 여유입니다. 더 작은 모델이 두 가지를
동시에 하는 데 *더 낫다*고 기대할 이유는 없습니다.

로보틱스에 대한 진짜 함의는 경고가 아니라 그 반대입니다: **S1은 텍스트-온리입니다.** 이미지도,
3D 메모리도 없고, subgoal당 한 번 호출이며, 프롬프트는 수천 토큰입니다. 로봇 위에서 돌 필요가
없습니다. 분리는 크기 요구를 분리해줍니다 — 온디바이스에 2–4B 정책, 오프보드에 더 큰 해소기 —
그리고 그것이 정확히 1-step이 금지하는 것입니다.

### 6.5 이 곱셈이 과대평가하는 것

- **카테고리는 인스턴스가 아니다.** 논문의 79.2%는 올바른 카테고리의 *아무* 물체에나 도달하면
  성공으로 셉니다. RefON은 *특정* 물체까지의 거리로 채점합니다. **balanced 세트 subgoal의
  48.9%가 특정한 이전 인스턴스에 대한 back reference**입니다. 카테고리만 넘기는 핸드오프는 그
  정보를 잃으므로 anchor A는 절반의 항목에 대해 낙관적입니다. 대신 인스턴스 정체성을 경계 너머로
  넘기면 S2는 더 이상 평범한 object-nav가 아닙니다.
- **독립 가정.** `S1 × S2`는 해소 실패와 네비게이션 실패가 무상관이라고 가정합니다. 어려운
  지시문은 찾기 어려운 물체와 함께 갈 개연성이 있고, 그렇다면 실제 값은 곱보다 낮아집니다.
- **표본 크기와 모델 불일치.** 13.5%는 n=37, 20.0%는 n=15(±20점)입니다. 그리고 13.5%는
  `qwen3-vl:30b`, 79.2%는 GPT-4o, S1 열은 또 다른 여섯 모델입니다.
- **오류가 되돌릴 수 없게 된다.** 1-step 에이전트는 원리상 씬을 본 뒤 오독을 수정할 수 있지만
  2-step은 못 합니다. 곱셈은 이를 모델링하지 않으며, 이는 2-step에 유리하게 작용합니다.
  (다만 9.1%라는 참조 행 수치는 그 수정 채널이 현재 거의 작동하지 않음을 시사합니다.)

### 6.6 텍스트-온리 단계가 할 수 없는 두 가지

**물체의 부재 확인.** `GA_absent_object`("Find the chandelier."인데 씬에 샹들리에가 없음)는
어떤 능력 수준에서도 텍스트로 답할 수 없습니다 — 부재는 *탐색의 결과*입니다. 해소기는 "샹들리에를
찾아라"까지만 낼 수 있고, 결론은 네비게이션에서 돌아와야 합니다. §1.2의 2턴 프로토콜이 바로 그
인정이며, referent probe는 `GA_absent_object`를 `new` + 카테고리로 채점함으로써(§1.1) 이
문제를 우회합니다 — 즉 **92.8%라는 헤드라인 수치에는 이 실패 모드가 아예 들어 있지 않습니다.**

**궤적 상대적 참조.** *"find another toilet, not the first one you found"* 같은 지시문은
배제 집합을 지시문 텍스트가 아니라 에이전트의 **실제 경로** 위에서 정의합니다. 텍스트-온리
해소기는 어느 변기에 먼저 도달했는지 알 수 없습니다.

RefON은 현재 이 케이스를 담고 있지 않습니다. 생성된 세트의 모든 참조 형태는 *지시문* 상대적입니다:

```
OR_post   "Find the 1st one again."
AR_post   "Go back to the previous one."  /  "Find the one before that."
AB_post   "Find A1."
```

서수가 지시문 시퀀스를 인덱싱하고 생성기가 그것을 정답 물체에 바인딩하므로, 지시문을 세는 것만으로
충분합니다 — 텍스트-온리 점수가 지금처럼 높은 이유의 일부가 이것입니다. 궤적 상대적 스타일
(가칭 `TR_post`)이 2-step 분리의 정직한 시험이 될 것이고, 이를 추가하는 것이 명백한 다음 데이터셋
변경입니다.

### 6.7 결론

유용한 아키텍처는 1-step도 2-step도 아니라 **얇은 상태 채널을 가진 2-step**입니다: 해소기가
지시문 히스토리에 더해 에이전트가 실제로 방문한 것의 압축된 텍스트 요약(object id, 카테고리, 방,
방문 순서)과 `not_found` 결과를 함께 받습니다. 여전히 텍스트-온리이고, 여전히 싸고, 여전히
오프보드 가능합니다 — 그리고 이것이 `run_refonbench_feasibility_nav.py`가 subgoal이 도달할 때마다
좌표를 공개하는 방식으로 이미 모델링하고 있는 것입니다.

---

## 7. 한계

### 7.1 thinking 모델은 있는 그대로 비교할 수 없다

qwen3.5는 모든 크기에서 토큰 예산 전체를 `reasoning`에 쓰고 빈 `content`를 반환합니다. 32768까지
시도한 어떤 예산에서도 그렇습니다. 보고된 qwen3.5 수치는 전부 `reasoning_effort=none`입니다.
`gemma4:26b`는 thinking이 켜져 있으므로 94.8%는 사다리의 동등 항목이 **아닙니다**.
`claude-haiku-4-5`는 `thinking`을 비활성화하고 돌려 qwen 조건과 일치하지만, 크기 비공개 API
모델이라 다른 이유로 사다리 밖입니다(§1.4).

### 7.2 신규 물체 subgoal의 약 9–13%는 텍스트로 답할 수 없다

이전 지시문이 이미 사용한 *카테고리*의 새 물체를 도입하는 subgoal("Find the cardboard box."인데
1번 지시문도 "Find the cardboard box."였던 경우)은 씬을 보지 않고는 back reference와 구별할 수
없습니다: balanced 세트에서 **49/533 (9%)**, long 세트에서 **54/403 (13%)**. 이것이 `S` 행에
도달 불가능한 천장을 만들고, 가장 강한 두 모델이 `S`에서 back reference보다 *낮은* 점수를 받는
이유를 설명합니다 — gemma4 91.8 대 96.8, haiku 91.4 대 93.6. 답이 유일하게 결정되는
back-reference 행에는 영향이 없고, 모든 모델에 동일하게 적용되므로 모델 비교는 유효합니다.

### 7.3 헤드라인 수치는 작은 모델에서 프롬프트에 민감하다

referent probe에서 답 스키마를 세 번 시도했습니다. 지시문 번호만 요구했을 때 qwen2.5vl:7b는
534개의 새 물체 중 231개에 대해 어떤 이전 지시문을 가리켰습니다. 같은 필드의 값 중 하나로
`"new"`를 제공했더니 373개의 back reference에 `"new"`라고 답했는데, 그중 279개는 여전히 진짜
referent의 카테고리를 맞게 말했습니다 — 즉 참조를 해소하고도 라벨만 잘못 붙인 것입니다. 채택한
스키마(라벨, 그리고 라벨이 `back_reference`일 때만 번호)는 두 결정을 분리해 두지만, feasibility
수치는 **같은 프롬프트 버전**으로 얻은 네비게이션 수치와만 비교 가능합니다.

### 7.4 단일 씬, 단일 시드

사용 가능한 goal 물체가 약 19개인 HM3D 씬 하나, 생성기 시드 하나, temperature 0. 카테고리
다양성이 제한적이고(20개) 에피소드 내 물체 재사용이 많으며, 그것이 §7.2를 지금 크기로 만드는
원인입니다.

---

## 8. 인프라 기록

실행 자체가 찾아낸 결함이 셋 있으며, 어떤 수치든 읽기 전에 알아둘 가치가 있습니다.

- **`max_tokens`를 넘긴 reasoning 모델은 예외 없이 빈 content를 반환합니다.** 클라이언트 기본값
  4096에서 gemma4는 200개 `all_at_once` 에피소드 중 61개에 아무것도 반환하지 않았고, 채점기는
  그것을 451개의 오답으로 읽었습니다 — 63.8%로 보고된 것이 실은 토큰 예산이었습니다. 이제 기록은
  `empty_response`를 `parse_failed`와 분리해 담고, 실행기가 결과 표 위에 경고를 출력합니다.
- **예산에 잘린 응답은 재시도할 수 없습니다** — 같은 프롬프트는 같은 방식으로 잘립니다. **연속**
  5회 절단이면 모델을 드롭하고 결과를 아예 쓰지 않으므로, 드롭된 모델을 나쁜 점수로 오인할 수
  없습니다. 누적으로 세었을 때는 0.3%의 산발적 절단률 때문에 멀쩡한 실행 둘을 버렸습니다.
- **`VLMClient`가 잘못된 편집 후 속성 절반을 조용히 잃었습니다.** 모든 요청이 `AttributeError`를
  던졌고, 60초 간격으로 다섯 번 재시도된 뒤, 요청이 단 한 번도 전송되지 않은 채 오답으로
  채점되었습니다. 1308개 질의를 6분에 처리하던 모델이 46분에 36개를 처리했습니다. 처리량이 한
  자릿수 배 이상 떨어진 실행은 수치를 읽기 전에 로그에서 오류를 확인해야 합니다.

---

## 9. 재현

```bash
# referent probe (누적 제시), balanced 세트
OLLAMA_MODEL=qwen3.5:9b python run_refonbench_feasibility.py \
    -cf cfg/eval_refonbench_feasibility.yaml --workers 4 --reasoning-effort none

# destination probe
OLLAMA_MODEL=qwen3.5:9b python run_refonbench_feasibility_nav.py \
    -cf cfg/eval_refonbench_feasibility_nav.yaml --workers 4 --reasoning-effort none

# 긴 에피소드 세트
OLLAMA_MODEL=qwen3.5:9b python run_refonbench_feasibility.py \
    -cf cfg/eval_refonbench_feasibility_long.yaml --workers 4 --reasoning-effort none

# 같은 세 probe를 Anthropic API로 (키는 ANTHROPIC_API_KEY에서 읽음)
VLM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-haiku-4-5 \
    python run_refonbench_feasibility.py \
    -cf cfg/eval_refonbench_feasibility.yaml --workers 4 --reasoning-effort none

# 표와 차트 (--merge-roles가 AB_pre / AR_pre를 S로 접음, 이 문서와 동일)
python scripts/compare_feasibility.py --merge-roles \
    results/exp_feasibility_refonbench_* --plot cmp_even.png
python scripts/compare_feasibility.py --merge-roles --nav \
    results/exp_feasibility_nav_refonbench_* --plot cmp_nav.png
python scripts/compare_probe_vs_nav.py --merge-roles --plot cmp_probe_vs_nav.png

# §6.1 — 1-step 파이프라인 자체의 점수, 네비게이션 실행의 pickle에서
python3 -c "import pickle; t=pickle.load(open(
    'results/exp_eval_refonbench_default_qwen3_vl_30b/success_by_task_0.0_1.0_0.pkl','rb'));
    print({k: (len(v), sum(v)) for k, v in t.items()})"
```

`compare_probe_vs_nav.py`는 `*_nothink_*` glob이 기본이므로 API 실행을 포함하려면
`--probe-glob` / `--nav-glob`을 넘기십시오. `compare_feasibility.py --plot`의 `--dataset`은
세트 루트가 아니라 shard 디렉터리 자체
(`RefONEpisodeGenerator/out/refon_example_long_dataset/v1/val/content/`)를 가리켜야 하며,
아니면 참조 거리 패널이 빈 채로 나옵니다.

모든 오답의 전체 대화 — 시스템 프롬프트, 사용자 프롬프트, 응답 — 는 각 결과 디렉터리의
`feasibility_failures_<mode>.log` / `feasibility_nav_failures.log`에 있고, 세 subgoal을 틀린
`all_at_once` 응답 하나가 세 벌이 아니라 한 개의 대화 기록이 되도록 묶여 있습니다.

### 산출물

| 무엇 | 어디 |
|---|---|
| 실행별 결과 | `results/exp_feasibility_*/` (17개 실행) |
| 1-step 파이프라인 실행 (§6.1) | `results/exp_eval_refonbench_default_qwen3_vl_30b/` |
| 차트 | `results/cmp_even.png`, `cmp_long.png`, `cmp_nav.png`, `cmp_probe_vs_nav.png`, `cmp_even_haiku.png` |
| 데이터셋 | `RefONEpisodeGenerator/out/refon_example_even_dataset/`, `refon_example_long_dataset/` |
| 생성기 설정 | `RefONEpisodeGenerator/configs/generator.example_even.json`, `generator.example_long.json` |
| 3D-Mem 논문 수치 (§6.2) | arXiv 2411.17735, Table 1과 3 |

---

## 10. 부록: 프롬프트 전문과 성공 응답

프롬프트는 **영문 그대로 전송됩니다.** 한글로 바꾸면 기존 17개 실행과 비교가 불가능해지므로
코드는 손대지 않았습니다. 아래 번역은 읽기용입니다.

### 10.1 referent probe — 시스템 프롬프트

```
You are analysing the instructions of an indoor object-search task. A robot is given a
sequence of instructions, one at a time, and each instruction sends it to exactly one
object in the house.
Instructions come in several forms:
  * a direct one names the object: "Find the toilet."
  * an alias reference points at an object that was named earlier: "Find A1.", where A1
    was bound by an earlier instruction ending in "Let's call it A1."
  * an ordinal reference counts over the instructions already given: "Find the 2nd one
    again." means the object of the 2nd instruction.
  * a relative reference points just before the latest visit: "Go back to the previous
    one." / "Find the one before that." / "Go back to the one before the last." all mean
    the object of the instruction *before the most recent one*, not the most recent one
    itself.
"Let's call it X" binds the name X to the object being found by that very instruction; it
is not a request to search for something called X.
An alias that was never bound refers to no object at all, and neither does an ordinal that
points past the instructions given so far.
You are not navigating and you cannot see the house. Your only job is to work out which
object each instruction is talking about, from the instructions themselves.
Answer with JSON only. No prose, no markdown fences.
```

**번역:**

> 당신은 실내 물체 탐색 과제의 지시문을 분석하고 있습니다. 로봇은 지시문을 한 번에 하나씩
> 순서대로 받으며, 각 지시문은 로봇을 집 안의 정확히 하나의 물체로 보냅니다.
> 지시문에는 여러 형태가 있습니다:
>   * **직접 지시**는 물체를 이름으로 부릅니다: "Find the toilet."
>   * **별칭 참조**는 앞서 이름 붙인 물체를 가리킵니다: "Find A1." — A1은 "Let's call it A1."로
>     끝나는 이전 지시문이 바인딩한 것입니다.
>   * **서수 참조**는 이미 주어진 지시문들을 셉니다: "Find the 2nd one again."은 2번 지시문의
>     물체를 뜻합니다.
>   * **상대 참조**는 가장 최근 방문의 바로 앞을 가리킵니다: "Go back to the previous one." /
>     "Find the one before that." / "Go back to the one before the last."는 모두 *가장 최근
>     지시문의 바로 앞* 지시문의 물체를 뜻하며, 가장 최근 것 자체가 아닙니다.
> "Let's call it X"는 바로 그 지시문이 찾고 있는 물체에 이름 X를 바인딩하는 것이지, X라는 것을
> 찾아달라는 요청이 아닙니다.
> 바인딩된 적 없는 별칭은 어떤 물체도 가리키지 않으며, 지금까지 주어진 지시문 범위를 넘는 서수도
> 마찬가지입니다.
> 당신은 네비게이션을 하지 않으며 집을 볼 수 없습니다. 당신의 유일한 임무는 각 지시문이 어떤
> 물체를 말하는지를 지시문 자체만으로 알아내는 것입니다.
> JSON만으로 답하십시오. 산문도, 마크다운 펜스도 쓰지 마십시오.

### 10.2 referent probe — 사용자 프롬프트와 성공 응답

실제 사례: episode 1, subgoal 3, `AR_post`, `claude-haiku-4-5`.

```
Instructions given to the robot so far, in order:
  1. "Find the cabinet."
  2. "Find the 1st one again. Let's call it A1."
  3. "Find the one before that."

Question: instruction 3 sends the robot to one object. Is that object one an earlier
instruction already went to, or a new one?

Reply with a single JSON object:
{"answer": "new" | "back_reference" | "no_object", "refers_to": <integer 1..3 or null>,
 "category": <string or null>, "reason": <string>}
  "answer": exactly one of
      "new"            -- no earlier instruction went to this object; the instruction
                          names a fresh one.
      "back_reference" -- an earlier instruction already went to this same object.
      "no_object"      -- the instruction points at nothing (an alias that was never
                          bound, an ordinal past the end of the list).
  "refers_to": only when "answer" is "back_reference" -- the number of the earlier
               instruction that went to the object. null otherwise.
  "category": the object category as a short noun phrase ("toilet", "cardboard box").
              Always name it -- for a back reference, repeat the category the earlier
              instruction used. Only "no_object" may leave it null.
  "reason": one short sentence.
```

**성공 응답(기록된 원문):**

```json
{
  "answer": "back_reference",
  "refers_to": 1,
  "category": "cabinet",
  "reason": "Instruction 3 says 'Find the one before that,' which is a relative reference
             pointing to the instruction before the most recent one (instruction 2),
             meaning instruction 1 which found the cabinet."
}
```

정답은 `refers_to=1`, `category=cabinet`. 까다로운 이유는 2번이 이미 1번을 되짚은 상태여서
"그 앞"이 **지시문 순서상의 앞**(1번)이지 방문 순서상의 앞이 아니라는 점입니다.

### 10.3 destination probe — 시스템 프롬프트

지시문 형태 설명은 §10.1과 같고, 앞부분이 다릅니다:

```
You are directing a robot through a house. It is given instructions one at a time, and
each instruction sends it to exactly one object.
You cannot see the house. The only thing you know about it is what the log of earlier
instructions tells you: once the robot has been to an object, that object's (x, y)
position is known, and it stays known for the rest of the episode. An object nobody has
been to yet has no known position, and you have no way to guess it.
When the robot is sent to explore, one of two things comes back: the object's position,
or `not_found`. `not_found` means no such object exists anywhere in this house, so that
instruction can never be carried out.
```

**번역:**

> 당신은 로봇을 집 안에서 지휘하고 있습니다. 로봇은 지시문을 한 번에 하나씩 받고, 각 지시문은
> 로봇을 정확히 하나의 물체로 보냅니다.
> 당신은 집을 볼 수 없습니다. 집에 대해 아는 것은 이전 지시문들의 로그가 알려주는 것뿐입니다:
> 로봇이 어떤 물체에 한 번 다녀오면 그 물체의 (x, y) 위치가 알려지고, 에피소드가 끝날 때까지
> 계속 알려진 상태로 남습니다. 아직 아무도 가보지 않은 물체는 알려진 위치가 없으며, 추측할
> 방법도 없습니다.
> 로봇을 탐색으로 보내면 둘 중 하나가 돌아옵니다: 물체의 위치, 또는 `not_found`.
> `not_found`는 이 집 어디에도 그런 물체가 없다는 뜻이며, 따라서 그 지시문은 결코 수행될 수
> 없습니다.

행동 스펙:

> - `"explore"` — 지시문이 로봇을 로그에 위치가 없는 물체로 보냄. 아무도 가본 적 없으니 찾아야 함.
> - `"(x, y)"` — 지시문이 로봇을 로그에 위치가 **있는** 물체로 되돌려 보냄. 그 위치를 **정확히
>   복사**할 것. 예: `"(3.2, -1.4)"`.
> - `"infeasible"` — 지시문을 아예 수행할 수 없음: 바인딩된 적 없는 별칭이거나, 목록 끝을 넘는
>   서수이거나, 이미 탐색이 `not_found`로 돌아온 경우.

### 10.4 destination probe — 성공 응답 세 가지

**좌표 복귀** (episode 3, subgoal 4, `OR_post`):

```
Log of the instructions given so far, and what came of each:
  1. "Find the door."               ->  explored, and found the door at (7.7, -5.7)
  2. "Find the 1st one again."      ->  went to (7.7, -5.7)
  3. "Find the one before that."    ->  went to (7.7, -5.7)

Now: instruction 4. "Find the 1st one again."
```
```json
{"action": "(7.7, -5.7)", "reason": "The 1st instruction found the door at (7.7, -5.7),
 so the robot returns to that known position."}
```

**부재 물체 2턴 프로토콜** (episode 1, subgoal 7, `GA_absent_object`). 로그에 6개 지시문이
쌓인 상태에서 `"Find the chandelier."`가 들어옵니다:

```json
턴 1:  {"action": "explore",    "reason": "The chandelier has not been visited before,
                                 so its position is unknown and must be explored."}

환경:  You sent the robot to explore. It searched and reported: not_found

턴 2:  {"action": "infeasible", "reason": "The chandelier has already been searched for
                                 and reported as not_found, so it cannot be found in
                                 this house."}
```

로그가 모델의 답이 아니라 실제로 일어난 일을 담는다는 점이 여기서 중요합니다(§1.2). 그래서
3번에서 틀려도 4번 이후가 연쇄로 무너지지 않고, 모든 턴이 독립 질의라 병렬 실행됩니다.
