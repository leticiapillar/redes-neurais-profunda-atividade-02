# Segmentação de Tumores Cerebrais em MRI com U-Net

- **Disciplina:** Redes Neurais Profundas — Atividade 02
- **Aluna:** Leticia Pillar Lisboa
- **Professor:** Felipe André Zeiser
- **Repositório:** https://github.com/leticiapillar/aprendizado-por-reforco-modelagem-mdp
- **Dataset:** LGG Segmentation Dataset — https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

---

## 1. Visão geral

Trabalho prático de visão computacional que implementa uma **U-Net** para segmentação
binária (tumor vs. fundo) em imagens de MRI cerebral, usando o **LGG Segmentation
Dataset** (`datasets/kaggle_3m`), disponível em
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation.

O notebook [`notebook/segmentacao_unet.ipynb`](notebook/segmentacao_unet.ipynb) percorre
as 5 etapas do exercício:

1. Carregamento e análise do dataset (divisão treino/val/teste, visualização inicial)
2. Pré-processamento (resize, normalização, augmentation)
3. Construção do modelo (arquitetura U-Net)
4. Treinamento e validação
5. Avaliação e interpretação dos resultados (quantitativa e qualitativa)

> **Dataset e resultados já incluídos no repositório.** Não é preciso baixar nada à parte
> (sem conta/API do Kaggle): as imagens+máscaras (`datasets/kaggle_3m/`), o checkpoint
> treinado (`outputs/checkpoints/best_model.pt`) e o notebook **já executado**, com todos
> os gráficos e métricas, estão versionados no repositório. Basta clonar e abrir o
> notebook para ver os resultados — não é necessário rodar nada. As instruções da seção 4
> servem para **reproduzir o experimento do zero**, caso desejado.

## 2. Resultados

Modelo treinado por 40 épocas (early stopping habilitado), imagens 128×128, ~47 min em CPU.

| Conjunto de teste       | Dice   | IoU    | Acurácia de pixel |
|-------------------------|--------|--------|--------------------|
| Geral (523 fatias)      | 0.6449 | 0.6122 | 0.9955             |
| Apenas com tumor (173)  | 0.7987 | 0.6930 | 0.9917             |

O melhor Dice de validação (fatias com tumor) foi **0.7660**. A métrica "apenas com
tumor" é mais informativa que a geral, já que fatias sem tumor inflam Dice/IoU de forma
trivial (o modelo acerta "tudo fundo" facilmente).

Curvas de treino e exemplos de predição estão em [`outputs/figures/`](outputs/figures/)
e reproduzidos dentro do notebook.

## 3. Estrutura do projeto

```
.
├── datasets/kaggle_3m/          # dataset original (imagem + máscara .tif por paciente)
├── notebook/
│   └── segmentacao_unet.ipynb   # notebook com as 5 etapas do exercício (já executado)
├── outputs/                     # artefatos gerados pelo treino/avaliação
│   ├── checkpoints/best_model.pt
│   ├── figures/                 # curvas de treino e predições de exemplo
│   ├── config.json              # hiperparâmetros e estatísticas de normalização
│   ├── history.json             # histórico de treino por época
│   ├── test_metrics.json        # métricas finais no conjunto de teste
│   ├── manifest.csv             # tabela imagem<->máscara
│   └── split_{train,val,test}.csv
├── src/
│   ├── manifest.py               # varre o dataset e monta a tabela imagem<->máscara
│   ├── splits.py                 # divisão treino/val/teste por paciente
│   ├── dataset.py                # Dataset PyTorch (resize, normalização, augmentation)
│   ├── model.py                  # arquitetura U-Net
│   ├── losses.py                 # BCE + Dice loss, métricas (Dice, IoU, acurácia)
│   ├── engine.py                 # loops de treino/avaliação
│   ├── train.py                  # script de treino completo (gera outputs/)
│   └── evaluate_test.py          # avaliação final no conjunto de teste
├── requirements.txt
└── README.md
```

## 4. Pré-requisitos

- Python 3.11 (o [`uv`](https://docs.astral.sh/uv/), abaixo, baixa essa versão
  automaticamente caso não exista no sistema).
- `uv` para criar o ambiente virtual e instalar as dependências:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Alternativa sem `uv`:
  ```bash
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
  ```
- **Não requer GPU** — todo o projeto foi desenvolvido e testado em CPU (o treino
  completo leva cerca de 45–50 min nessas condições).

## 5. Como executar localmente

### 5.1 Criar o ambiente

```bash
uv venv -p 3.11 .venv

# PyTorch é instalado à parte, a partir do índice CPU-only do PyTorch (build bem menor
# que a versão padrão do PyPI, que assume GPU/CUDA):
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Demais dependências (numpy, pillow, matplotlib, scikit-learn, pandas, tqdm, jupyter, ...):
uv pip install --python .venv/bin/python -r requirements.txt
```

### 5.2 Visualizar os resultados já prontos

Não é necessário rodar nada: basta abrir `notebook/segmentacao_unet.ipynb` (já
executado, com todas as saídas e gráficos) no Jupyter, VS Code ou similar.

```bash
.venv/bin/jupyter lab notebook/segmentacao_unet.ipynb
```

### 5.3 Reproduzir o experimento do zero (opcional)

Todos os artefatos abaixo já estão versionados em `outputs/` e no notebook executado; só
é necessário rodar estes passos para **reproduzir** o experimento (ex.: treinar de novo,
mudar hiperparâmetros). Rodar cada script sobrescreve os arquivos correspondentes em
`outputs/`.

```bash
# 1. Treino (gera outputs/checkpoints/best_model.pt, curvas, histórico) — ~45-50 min em CPU
.venv/bin/python src/train.py

# 2. Avaliação final no conjunto de teste + figuras — poucos segundos
.venv/bin/python src/evaluate_test.py

# 3. Reexecuta o notebook com as 5 etapas do exercício (usa os artefatos gerados acima,
#    não retreina) — ~1-2 min
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook/segmentacao_unet.ipynb
```

## 6. Decisões principais

- **Divisão por paciente** (não por fatia) para treino/val/teste, evitando vazamento
  entre fatias vizinhas do mesmo exame.
- **Resize para 128×128** e normalização por canal (estatísticas calculadas no treino).
- **BCE + Dice Loss** combinadas, adequadas ao desbalanceamento entre fundo e região de
  tumor.
- **Early stopping** + dropout + weight decay + augmentation como estratégias de
  regularização.
- Métricas de teste reportadas tanto no conjunto geral quanto **apenas nas fatias com
  tumor** (mais informativo, já que fatias vazias inflam Dice/IoU triviais).
