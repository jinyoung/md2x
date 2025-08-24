# Markdown to HTML Converter with Mermaid Support

이 프로젝트는 복잡한 Markdown 파일에 포함된 Mermaid 다이어그램을 HTML 형태로 변환하는 유틸리티입니다.

## 생성된 파일들

### 입력 파일
- `sample.md` - 원본 Markdown 파일 (8개의 Mermaid 다이어그램 포함)

### 변환 결과물

1. **SVG 임베드 방식 (mermaid-cli 사용)**
   - `sample_converted.md` - Mermaid 다이어그램이 SVG 이미지 링크로 대체된 Markdown 파일
   - `sample_converted-[1-8].svg` - 각 Mermaid 다이어그램이 변환된 SVG 파일들
   - `sample_static.html` - SVG가 임베드된 정적 HTML 파일

2. **동적 렌더링 방식 (Mermaid.js 사용)**
   - `sample_dynamic.html` - 브라우저에서 실시간으로 Mermaid 다이어그램을 렌더링하는 HTML 파일

3. **인라인 SVG 방식**
   - `sample_inline.html` - SVG 코드가 HTML 내에 직접 임베드된 자체 완결형 파일

4. **DOCX 방식 (새로 추가!) 📄**
   - `sample_test.docx` - 기본 DOCX 변환 결과
   - `sample_professional.docx` - 고급 스타일링이 적용된 전문적인 DOCX 문서

### 변환 도구
- `convert_to_html.py` - Python 기반 HTML 변환 스크립트 (라이브러리 함수들)
- `md2html_converter.py` - HTML 변환 명령줄 인터페이스 (추천)
- `md2docx_converter.py` - DOCX 변환 명령줄 인터페이스 (기본)
- `md2docx_advanced.py` - DOCX 변환 고급 버전 (전문적 스타일링) ⭐

## 변환 과정

### 1단계: mermaid-cli 설치 및 SVG 변환
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i sample.md -o sample_converted.md
```

### 2단계: HTML 변환
```bash
pip install markdown

# 방법 1: 통합 스크립트 (추천)
python3 convert_to_html.py

# 방법 2: 명령줄 도구 (개별 변환)
python3 md2html_converter.py sample.md --mode inline    # 인라인 SVG
python3 md2html_converter.py sample.md --mode dynamic   # 동적 렌더링
python3 md2html_converter.py sample_converted.md --mode static  # 정적 SVG

# 방법 3: DOCX 변환 (새로 추가!)
python3 md2docx_converter.py sample.md                  # 기본 DOCX
python3 md2docx_advanced.py sample.md                   # 고급 전문적 DOCX (추천)
```

## 결과물 특징

### Static HTML (`sample_static.html`)
- **장점**: 
  - 오프라인에서도 동작 (SVG 파일이 로컬에 있음)
  - 빠른 로딩 속도
  - 정적 호스팅에 적합
- **단점**: 
  - SVG 파일들이 별도로 필요
  - 다이어그램 수정 시 재변환 필요

### Dynamic HTML (`sample_dynamic.html`)
- **장점**: 
  - 단일 HTML 파일로 완결
  - 브라우저에서 고품질 렌더링
  - 반응형 다이어그램
- **단점**: 
  - 인터넷 연결 필요 (CDN 의존)
  - 초기 로딩 시간이 약간 길 수 있음

### Inline SVG HTML (`sample_inline.html`) ⭐ **추천**
- **장점**: 
  - **완전한 자체 완결형** (외부 파일이나 인터넷 연결 불필요)
  - **오프라인에서 완벽 동작**
  - **고품질 벡터 그래픽**
  - **빠른 로딩** (모든 리소스가 단일 파일에 포함)
- **단점**: 
  - 파일 크기가 상대적으로 큼 (약 200KB)
  - 생성 시간이 약간 더 오래 걸림

### DOCX Documents 📄 **새로 추가!**

#### Basic DOCX (`sample_test.docx`)
- **장점**: 
  - **Word와 완전 호환**
  - **편집 가능한 문서 형식**
  - **오프라인 완벽 지원**
  - **Mermaid 다이어그램이 고품질 이미지로 포함**
- **단점**: 
  - 기본적인 스타일링만 적용

#### Professional DOCX (`sample_professional.docx`) ⭐ **추천**
- **장점**: 
  - **전문적인 문서 스타일링**
  - **컬러 헤딩과 타이포그래피**
  - **표와 수식 포맷팅**
  - **고해상도 다이어그램** (2x 스케일링)
  - **코드 블록 전용 스타일**
  - **Microsoft Word, Google Docs 등에서 완벽 지원**
- **단점**: 
  - 파일 크기가 가장 큼 (약 850KB)

## 포함된 기능

- ✅ **Mermaid 다이어그램 렌더링** (flowchart, sequence, pie, timeline 등)
- ✅ **수학 공식 렌더링** (MathJax 사용)
- ✅ **문법 하이라이팅** (Prism.js 사용)
- ✅ **반응형 디자인**
- ✅ **현대적인 스타일링**
- ✅ **테이블 지원**
- ✅ **목차 자동 생성**

## 브라우저에서 확인

생성된 HTML 파일들을 브라우저에서 열어 확인할 수 있습니다:

```bash
open sample_dynamic.html      # 동적 Mermaid 렌더링
open sample_static.html       # 정적 SVG 임베드  
open sample_inline.html       # 인라인 SVG (추천)
open sample_test.docx          # 기본 DOCX
open sample_professional.docx  # 전문적 DOCX (추천)
```

## 포함된 Mermaid 다이어그램 종류

1. **시스템 아키텍처 개요** (flowchart)
2. **마이크로서비스 아키텍처** (graph)
3. **모델 학습 파이프라인** (flowchart)
4. **실시간 데이터 스트리밍** (sequence diagram)
5. **배치 처리 아키텍처** (graph)
6. **모델 성능 비교** (pie chart)
7. **리소스 사용량** (pie chart)
8. **시스템 발전 로드맵** (timeline)

이 변환 도구는 기술 문서, 설계 문서, 튜토리얼 등 복잡한 다이어그램이 포함된 Markdown 문서를 웹에서 보기 좋은 HTML 형태로 변환하는 데 유용합니다.

## 🚀 빠른 시작

**가장 추천하는 사용법 (Professional DOCX):**

```bash
# 1. 필수 패키지 설치
npm install -g @mermaid-js/mermaid-cli
pip install python-docx pillow markdown

# 2. 전문적인 DOCX 문서 생성
python3 md2docx_advanced.py your_markdown_file.md

# 3. 생성된 DOCX 파일 열기
open your_markdown_file_advanced.docx
```

**HTML 버전 (웹용):**

```bash
# 1. 필수 패키지 설치
npm install -g @mermaid-js/mermaid-cli
pip install markdown

# 2. 자체 완결형 HTML 생성
python3 md2html_converter.py your_markdown_file.md --mode inline

# 3. 생성된 HTML 파일 열기
open your_markdown_file_inline.html
```

### 출력 형식별 특징 요약

| 형식 | 파일 크기 | 오프라인 | 편집성 | 품질 | 호환성 | 추천도 |
|------|-----------|----------|--------|------|--------|--------|
| **DOCX Professional** | 850KB | ✅ | ✅ | ⭐⭐⭐ | ⭐⭐⭐ | 🥇 **최고** |
| HTML Inline SVG | 200KB | ✅ | ❌ | ⭐⭐⭐ | ⭐⭐ | 🥈 |
| **DOCX Basic** | 430KB | ✅ | ✅ | ⭐⭐ | ⭐⭐⭐ | 🥉 |
| HTML Dynamic | 46KB | ❌ | ❌ | ⭐⭐⭐ | ⭐⭐ | 4️⃣ |
| HTML Static | 42KB | ✅ | ❌ | ⭐⭐ | ⭐⭐ | 5️⃣ |
