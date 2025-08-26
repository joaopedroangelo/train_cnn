# CNN Train

Este repositório contém dois treinamentos de Redes Neurais Convolucionais (CNNs) desenvolvidas em **PyTorch**.  

Os modelos foram treinados em dois datasets diferentes, com diferentes propósitos de classificação:  
- **HAM10000** – Classificação em **sete classes** de tipos de lesões de pele.
- **Malignant vs Benign** – Classificação binária entre **lesão maligna** e **benigna**.

---
## Datasets

- [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  

- [Skin Cancer - Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)

> Os datasets são mantidos em `.zip` no repositório. O conteúdo extraído deve ser colocado na pasta `dataset/` correspondente.

> Baixe o .zip do HAM10000 em: [Link](https://drive.google.com/file/d/1Xr1WvkOyxuigPX8eEwl6XUdF0hZvIsKs/view?usp=sharing)

> No link acima, você encontra o dataset já separado nas devidas classes e pronto pro treinamento.

---
## Instalação

Clone o repositório:

```bash
git clone https://github.com/seuusuario/skin-cancer-classification.git
cd skin-lesion-classification
````

Instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

---
## Treinamento

### HAM10000 – Classificação em Sete Classes

```bash
cd HAM10000
python3 train.py
```

### Malignant vs Benign – Classificação Binária

```bash
cd Malignant_vs_Benign
python3 train.py
```

---
## Resultados

* **HAM10000**: Treina uma CNN personalizada para prever uma de sete classes.
* **Malignant vs Benign**: Treina uma CNN para prever se a lesão é maligna ou benigna.

As métricas utilizadas incluem:

* Accuracy
* Precision
* Recall
* F1-Score

---
## Predição em Imagens Individuais

Ambos os scripts possuem função para predição de imagens únicas.
Exemplo:

```python
predict_image("./dataset/test/akiec/ISIC_0024339.jpg", transform)
```

---
## Licença

Uso educacional e de pesquisa. Os datasets são de uso público sob as regras do Kaggle.
