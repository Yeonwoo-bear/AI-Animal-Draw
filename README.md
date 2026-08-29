# AI ANIMAL DRAW v14

부스용 1회 플레이 버전입니다.

## 게임 흐름
1. 학과 / 닉네임 / 전화번호 뒤 4자리 입력
2. 카메라 연결
3. 랜덤 동물 1개 출제
4. 최대 60초 동안 SPACE로 DRAW ON/OFF
5. 실시간 Top 3 확인
6. `그림 제출` 또는 60초 종료 시 최종 분석
7. 제시 동물의 AI 일치도(확률)로 랭킹 저장
8. 동점이면 제출 시간이 더 짧은 참가자가 우선

## 중요: 모델 파일 넣기
이 ZIP에는 개인 학습 모델 바이너리를 포함하지 않습니다.
기존 v13에서 사용 중인 고정 입력 shape 모델 파일을 아래 위치로 복사하세요.

`public/model/animal_draw_v11.tflite`

즉 구조가 다음과 같아야 합니다.

```
public/
  model/
    animal_draw_v11.tflite
    labels.json
```

## 실행
```bash
npm install
npm run dev
```

## 배포
```bash
git add .
git commit -m "AI Animal Draw v14"
git push
```
Vercel이 기존 GitHub 저장소와 연결되어 있다면 자동 재배포됩니다.

## 랭킹 저장 방식
현재 `localStorage` 방식입니다. 부스 한 대의 PC에서는 전체 참가자 랭킹이 누적됩니다.
다른 휴대폰/PC에서 접속한 기록까지 하나로 합치려면 Supabase 같은 공용 DB 연결이 필요합니다.

## v14 변경점
- 45초 → 60초
- 정답 자동 종료 제거
- 한 사람당 랜덤 동물 1회
- 학과 / 닉네임 / 전화번호 뒤 4자리 입력
- 최종 목표 동물 확률을 `AI 일치도`로 사용
- 일치도 높은 순 랭킹, 동점 시 빠른 제출 우선
- 실시간 Top 3는 유지
- 제출 버튼 추가
- 60초가 끝나면 자동 제출
- classifier의 렌더링 ensemble을 5개로 확대해 Air Drawing 입력 변동에 조금 더 강하게 조정

## AI 정확도에 대해
v14는 전처리와 test-time ensemble을 개선했지만, 모델 자체가 QuickDraw 기반이라 실제 Air Drawing과 데이터 분포 차이가 남아 있을 수 있습니다. 정확도를 크게 올리려면 다음 단계에서 웹 입력과 같은 stroke 렌더링 방식으로 학습 데이터를 재구성하고 재학습하는 것을 권장합니다.
