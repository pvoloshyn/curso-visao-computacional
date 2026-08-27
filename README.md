# 👁️ Visão Computacional Aplicada à Hiperautomação

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/01_01_lendo_mostrando_salvando_imagens.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/uso-educacional-lightgrey)

Material utilizado no curso **Visão Computacional Aplicada à Hiperautomação**.

O objetivo deste curso é apresentar os principais conceitos e técnicas de Visão Computacional utilizados em iniciativas de automação inteligente, **OCR**, **IDP** (Intelligent Document Processing), **detecção de anomalias**, **VLMs** (Vision Language Models), **rastreamento de objetos** e **reconhecimento facial**, combinando teoria e prática através de notebooks executáveis no Google Colab.

---

## 📑 Sumário

- [Como montar o ambiente](#-como-montar-o-ambiente)
- [Chaves de API utilizadas no curso](#-chaves-de-api-utilizadas-no-curso)
- [Estrutura das aulas](#-estrutura-das-aulas)
  - [Aula 01 — Fundamentos de Imagens](#aula-01--fundamentos-de-imagens)
  - [Aula 02 — Inteligência Visual](#aula-02--inteligência-visual)
  - [Aula 03 — Leitura Inteligente de Documentos](#aula-03--leitura-inteligente-de-documentos)
  - [Aula 04 — Detecção de Anomalias](#aula-04--detecção-de-anomalias)
  - [Aula 05 — Rastreamento e Reconhecimento](#aula-05--rastreamento-e-reconhecimento)
- [Estrutura do repositório](#-estrutura-do-repositório)
- [Público-alvo](#-público-alvo)

---

## 🚀 Como montar o ambiente

Você pode acompanhar o curso de duas formas: **direto no navegador**, sem instalar nada, ou **na sua máquina local**.

### Opção 1 — Google Colab (recomendado)

Não é necessário instalar absolutamente nada. Basta clicar no botão **Open in Colab** de qualquer aula abaixo.

Cada notebook já contém, logo na primeira célula de código, um bloco que detecta o ambiente Colab, clona este repositório e instala as dependências específicas daquela aula automaticamente:

```python
from IPython import get_ipython
if 'google.colab' in str(get_ipython()):
    print("Preparando ambiente Google Colab")
    !pip install opencv-python==5.0.0.93
    !pip install opencv-contrib-python==5.0.0.93
    !git clone https://github.com/pvoloshyn/curso-visao-computacional.git
    %cd curso-visao-computacional
```

Basta executar as células em ordem — imagens, datasets e modelos usados em cada aula já vêm junto do `git clone`.

### Opção 2 — Ambiente local

Pré-requisitos: **Python 3.10+** e `git` instalados.

```bash
# 1. Clone o repositório
git clone https://github.com/pvoloshyn/curso-visao-computacional.git
cd curso-visao-computacional

# 2. Crie e ative um ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie o Jupyter
jupyter notebook
```

O arquivo [requirements.txt](requirements.txt) reúne todas as bibliotecas usadas ao longo do curso (`opencv-python`, `ultralytics`, `easyocr`, `landingai-ade`, `openai`, `anomalib`, `trackers`, `insightface`, `mediapipe`, entre outras). Algumas aulas têm dependências específicas — se preferir instalar sob demanda, siga o `!pip install` indicado no topo de cada notebook.

> 💡 As pastas `imagens/`, `datasets/` e `modelos/` contêm os arquivos de apoio (fotos, datasets e checkpoints de modelos) usados pelos notebooks e já fazem parte do repositório.

---

## 🔑 Chaves de API utilizadas no curso

Algumas aulas usam serviços externos de IA. As chaves são sempre solicitadas interativamente (via `getpass`) dentro do próprio notebook — nenhuma chave precisa ser configurada previamente no ambiente.

| Serviço | Usado em | Como obter |
|---|---|---|
| **OpenRouter** | `03_03_vlm.ipynb`, `04_02_anomalia_vlm.ipynb` | Crie uma conta em [openrouter.ai](https://openrouter.ai/) e gere uma chave em [Keys](https://openrouter.ai/workspaces/default/keys). Há modelos gratuitos disponíveis. |
| **LandingAI (ADE)** | `03_02_idp.ipynb` | Crie uma conta em [ade.landing.ai](https://ade.landing.ai/) e gere uma chave na seção **API Keys** do dashboard. |

---

## 📚 Estrutura das aulas

### Aula 01 — Fundamentos de Imagens

| Notebook | Descrição | <div style="width: 120px;" /> |
|---|---|---|
| 01.01 — Lendo, Mostrando e Salvando Imagens | Introdução ao OpenCV e manipulação básica de imagens. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/01_01_lendo_mostrando_salvando_imagens.ipynb) |
| 01.02 — Imagens Coloridas | Espaços de cores, canais RGB e conversões. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/01_02_imagens_coloridas.ipynb) |
| 01.03 — Histogramas | Análise de distribuição de pixels e contraste. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/01_03_histogramas.ipynb) |
| 01.04 — Manipulação Básica | Redimensionamento, recorte, rotações e transformações. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/01_04_manipulacao_basica.ipynb) |
| 01.05 — Ajustes, Filtros e Efeitos | Filtros espaciais, suavização e melhorias visuais. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/01_05_ajustes_filtros_efeitos.ipynb) |
| 01. Exercícios | Exercícios para reforçar o entendimento da aula. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/01_exercicios.ipynb) |

### Aula 02 — Inteligência Visual

| Notebook | Descrição | <div style="width: 120px;" /> |
|---|---|---|
| 02.01 — Classificação de Imagens | Módulo DNN do OpenCV, formato ONNX e classificação com MobileNetV2 e EfficientNet. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/02_01_classificacao.ipynb) |
| 02.02 — Detecção e Segmentação de Objetos | Detecção, segmentação e vocabulário aberto com a família YOLO e SAM. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/02_02_deteccao_segmentacao_objetos.ipynb) |
| 02.03 — Detecção e Segmentação com RF-DETR | Detecção, segmentação de objetos usando RF-DETR. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/02_03_rf_detr.ipynb) |
| 02. Exercícios | Exercícios para reforçar o entendimento da aula. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/02_exercicios.ipynb) |

### Aula 03 — Leitura Inteligente de Documentos

| Notebook | Descrição | <div style="width: 120px;" /> |
|---|---|---|
| 03.01 — OCR | Extração de texto de imagens e documentos com EasyOCR, incluindo pré-processamento de scans ruins. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/03_01_ocr.ipynb) |
| 03.02 — IDP (Intelligent Document Processing) | Transformando documentos em dados estruturados com a plataforma LandingAI (ADE). | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/03_02_idp.ipynb) |
| 03.03 — VLM (Vision Language Models) | Modelos multimodais via OpenRouter para extrair e interpretar informações de imagens e documentos. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/03_03_vlm.ipynb) |
| 03. Exercícios | Exercícios para reforçar o entendimento da aula. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/03_exercicios.ipynb) |

### Aula 04 — Detecção de Anomalias

| Notebook | Descrição | <div style="width: 120px;" /> |
|---|---|---|
| 04.01 — Detecção de Anomalias | Modelo EfficientAD (`anomalib`) treinado só com imagens normais para detectar defeitos, com heatmap e máscara de anomalia. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/04_01_deteccao_anomalias.ipynb) |
| 04.02 — Detecção de Anomalias com VLM | Combina o EfficientAD (localização precisa) com um VLM (laudo técnico em linguagem natural), incluindo saída estruturada em JSON. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/04_02_anomalia_vlm.ipynb) |
| 04. Exercícios | Exercícios para reforçar o entendimento da aula. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/04_exercicios.ipynb) |

### Aula 05 — Rastreamento e Reconhecimento

| Notebook | Descrição | <div style="width: 120px;" /> |
|---|---|---|
| 05.01 — Rastreamento de Objetos com YOLO + ByteTrack | Rastreamento de objetos em vídeo combinando os detectores YOLO com o `ByteTrack`, via `ultralytics`, atribuindo um ID persistente a cada objeto entre os quadros. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/05_01_rastreamento_objetos.ipynb) |
| 05.02 — Rastreamento de Objetos com a Biblioteca `trackers` da Roboflow | Rastreamento com a biblioteca `trackers`, da Roboflow, que separa o detector do rastreador, incluindo o `MotionEstimator` para compensar o movimento da câmera. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/05_02_rastreamento_objetos_roboflow.ipynb) |
| 05.03 — Detecção de Faces com a InsightFace | Detecção facial especializada com a `insightface`: localização, pontos de referência (*landmarks*) e estimativa de idade/gênero. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/05_03_deteccao_faces.ipynb) |
| 05.04 — Reconhecimento de Faces com a InsightFace | Reconhecimento facial com embeddings do `ArcFace` (`buffalo_l`): comparação de rostos e verificação de identidade 1:1. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/05_04_reconhecimento_faces.ipynb) |
| 05.05 — Pose Estimation com o MediaPipe | Estimativa de pose humana com o MediaPipe: localização do esqueleto (landmarks de ombros, cotovelos, quadris, joelhos etc.) para análise de postura e reconhecimento de gestos. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/05_05_pose_estimation.ipynb) |
| 05. Exercícios | Exercícios para reforçar o entendimento da aula. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvoloshyn/curso-visao-computacional/blob/main/05_exercicios.ipynb) |

---

## 🗂️ Estrutura do repositório

```
curso-visao-computacional/
├── 01_01...05_...ipynb   # Notebooks das aulas (numerados por aula.tópico)
├── imagens/              # Imagens de apoio, organizadas por aula (01/, 02/, 03/, 04/, 05/)
├── datasets/             # Datasets usados na Aula 04 (MVTec AD, Imagenette)
├── modelos/              # Checkpoints/pesos de modelos pré-treinados
├── scripts/              # Scripts auxiliares (ex.: download de datasets e treino da Aula 04)
└── requirements.txt      # Dependências Python do curso
```

---

## 🎯 Público-alvo

- Profissionais de Automação e Hiperautomação
- Desenvolvedores RPA
- Cientistas e Analistas de Dados
- Profissionais de IA Generativa
- Estudantes interessados em Visão Computacional
