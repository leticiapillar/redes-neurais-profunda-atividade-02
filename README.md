# Segmentação de Tumores Cerebrais em MRI com U-Net

- Aluna: Leticia Pillar Lisboa
- Professor: Felipe André Zeiser


Trabalho prático de visão computacional: segmentação binária (tumor vs. fundo) em imagens
de MRI cerebral, usando U-Net, no dataset **LGG Segmentation Dataset** (`datasets/kaggle_3m`).

> **Dataset e resultados já incluídos no repositório.** Não é preciso baixar nada à parte
> (sem conta/API do Kaggle): as imagens+máscaras (`datasets/kaggle_3m/`), o checkpoint
> treinado (`outputs/checkpoints/best_model.pt`) e o notebook **já executado**, com todos os
> gráficos e métricas (`notebook/segmentacao_unet.ipynb`), estão versionados no repositório.
> Basta clonar e abrir o notebook para ver os resultados — não é necessário rodar nada.
> As instruções abaixo servem para **reproduzir o experimento do zero**, caso desejado.

## Pré-requisitos

- Python 3.11 (o `uv`, abaixo, baixa essa versão automaticamente caso não exista no sistema).
- [`uv`](https://docs.astral.sh/uv/) para criar o ambiente virtual e instalar as dependências:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  (alternativa sem `uv`: `python3.11 -m venv .venv && source .venv/bin/activate && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`)
- **Não requer GPU** — todo o projeto foi desenvolvido e testado em CPU (o treino completo
  leva cerca de 45–50 min nessas condições; veja estimativas por etapa mais abaixo).

## Estrutura

```
datasets/kaggle_3m/        dataset original (imagem + máscara .tif por paciente)
src/
  manifest.py               varre o dataset e monta a tabela imagem<->máscara
  splits.py                 divisão treino/val/teste por paciente
  dataset.py                Dataset PyTorch (resize, normalização, augmentation)
  model.py                  arquitetura U-Net
  losses.py                 BCE + Dice loss, métricas (Dice, IoU, acurácia)
  engine.py                 loops de treino/avaliação
  train.py                  script de treino completo (gera outputs/)
  evaluate_test.py          avaliação final no conjunto de teste
notebook/segmentacao_unet.ipynb   notebook com as 5 etapas do exercício
outputs/                    checkpoints, curvas, métricas e figuras gerados pelo treino
```

## Ambiente

```bash
uv venv -p 3.11 .venv

# PyTorch é instalado à parte, a partir do índice CPU-only do PyTorch (build bem menor
# que a versão padrão do PyPI, que assume GPU/CUDA):
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Demais dependências (numpy, pillow, matplotlib, scikit-learn, pandas, tqdm, jupyter, ...):
uv pip install --python .venv/bin/python -r requirements.txt
```

## Execução (reprodução do zero — opcional)

Todos os artefatos abaixo já estão versionados em `outputs/` e no notebook executado; só é
necessário rodar estes passos para **reproduzir** o experimento (ex.: treinar de novo,
mudar hiperparâmetros). Rodar cada script sobrescreve os arquivos correspondentes em `outputs/`.

```bash
# 1. Treino (gera outputs/checkpoints/best_model.pt, curvas, histórico) — ~45-50 min em CPU
.venv/bin/python src/train.py

# 2. Avaliação final no conjunto de teste + figuras — poucos segundos
.venv/bin/python src/evaluate_test.py

# 3. Reexecuta o notebook com as 5 etapas do exercício (usa os artefatos gerados acima,
#    não retreina) — ~1-2 min
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook/segmentacao_unet.ipynb
```

Para apenas **visualizar** o notebook já executado (sem rodar nada), abra
`notebook/segmentacao_unet.ipynb` no Jupyter, VS Code ou similar.

## Decisões principais

- **Divisão por paciente** (não por fatia) para treino/val/teste, evitando vazamento entre
  fatias vizinhas do mesmo exame.
- **Resize para 128×128** e normalização por canal (estatísticas calculadas no treino).
- **BCE + Dice Loss** combinadas, adequadas ao desbalanceamento entre fundo e região de tumor.
- **Early stopping** + dropout + weight decay + augmentation como estratégias de regularização.
- Métricas de teste reportadas tanto no conjunto geral quanto **apenas nas fatias com tumor**
  (mais informativo, já que fatias vazias inflam Dice/IoU triviais).
