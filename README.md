# 배포용 패키지

이 폴더 통째로 GitHub 레포에 올리고 Streamlit Community Cloud에 연결하면 됩니다.

## 폴더 구성

```
deploy_package/
  franchise_agent/
    app.py                 # Streamlit 메인 앱 (Streamlit Cloud에서 "Main file path"로 지정)
    src/                   # 에이전트 코드 전체
    franchise.db           # 정보공개서 데이터 (SQLite, 로컬 MySQL 뷰를 그대로 덤프한 파일)
  brand_recommender_runtime/
    brand_recommender.py
    artifacts/
      brand_ranker.cbm     # 학습된 추천 모델
      metadata.joblib      # 브랜드데이터.csv 기반 브랜드 정보 (CSV 원본은 안 올려도 됨)
  fonts/
    NanumGothic.ttf        # PDF/차트용 한글 폰트 (SIL Open Font License, 자유 재배포 가능)
    NanumGothicBold.ttf
  requirements.txt
  .streamlit/
    secrets.toml.example   # 참고용 — 실제 값은 Streamlit Cloud 대시보드에서 입력
```

## DB에 대해

정보공개서 데이터는 읽기 전용·소규모(문서 733개)라서 별도 클라우드 DB 없이
`franchise_agent/franchise.db`(SQLite)로 통째로 번들했습니다. 로컬 MySQL의
`v_agent_*` 뷰 7개를 그대로 덤프한 파일이라 앱 코드(`src/db.py`) 수정 없이
그대로 동작합니다. 클라우드 DB 계정을 따로 만들 필요가 없습니다.

데이터가 나중에 바뀌면(신규 브랜드 추가 등) 로컬에서 다시 덤프해서 이 파일을
교체하고 재배포하면 됩니다.

## 배포 순서

1. 이 폴더(`deploy_package`)를 GitHub 레포에 push
2. https://share.streamlit.io 에서 새 앱 생성, 레포 연결
3. **Main file path**: `franchise_agent/app.py`
4. **Secrets**: 앱 설정 > Secrets 메뉴에 `.streamlit/secrets.toml.example` 내용을 참고해서
   `OPENAI_API_KEY`, `OPENAI_MODEL` 값만 입력 (DB는 이미 리포에 포함되어 있어 별도 입력 불필요)
5. Deploy

## 참고

- 폰트는 나눔고딕으로 이미 바꿔뒀습니다(맑은 고딕은 마이크로소프트 라이선스라 공개 레포에
  올리면 안 되고, 클라우드 Linux 서버엔 애초에 없음).
- `brand_recommender_runtime`은 원본 폴더 구조를 그대로 유지했습니다.
