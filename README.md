# Segmentação de Tumores Cerebrais em MRI com U-Net

Trabalho prático de visão computacional: segmentação binária (tumor vs. fundo) em imagens
de MRI cerebral, usando U-Net, no dataset **LGG Segmentation Dataset** (`datasets/kaggle_3m`).

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
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

## Execução

```bash
# 1. Treino (gera outputs/checkpoints/best_model.pt, curvas, histórico)
.venv/bin/python src/train.py

# 2. Avaliação final no conjunto de teste + figuras
.venv/bin/python src/evaluate_test.py

# 3. Notebook com as 5 etapas do exercício (usa os artefatos gerados acima)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook/segmentacao_unet.ipynb
```

## Decisões principais

- **Divisão por paciente** (não por fatia) para treino/val/teste, evitando vazamento entre
  fatias vizinhas do mesmo exame.
- **Resize para 128×128** e normalização por canal (estatísticas calculadas no treino).
- **BCE + Dice Loss** combinadas, adequadas ao desbalanceamento entre fundo e região de tumor.
- **Early stopping** + dropout + weight decay + augmentation como estratégias de regularização.
- Métricas de teste reportadas tanto no conjunto geral quanto **apenas nas fatias com tumor**
  (mais informativo, já que fatias vazias inflam Dice/IoU triviais).
