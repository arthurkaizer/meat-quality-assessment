# Passo a passo para rodar o projeto

### Imagens

Necessário descompactar a pasta de imagens para a pasta **assets** na raiz do projeto. É necessário também separar os arquivos por classes, com imagens de carnes boas em uma pasta e as imagens de carnes ruins em outra pasta, ambas dentro da pasta **assets**. Os nomes das pastas podem ser alterados no arquivo **configs.py**

### Criar o ambiente virtual:

`python -m venv venv`

### Ativar o ambiente virtual:

- No windows: `.\venv\Scripts\activate`
- No linux: `source venv/bin/activate`

Se o ambiente for ativado corretamente, talvez você terá um indicativo visual, por exemplo:
  `(venv) C:\Users... `

### Instalar dependências do projeto:

`pip install -r requirements.txt `

### Rodar o projeto:

- No windows: `python src\main.py`
- No linux: `python src/main.py`
