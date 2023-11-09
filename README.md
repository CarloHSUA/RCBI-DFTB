# RCBI-DFTB

## Requisitos
1. Tener instalada la versión de 11.8 de cuda, si no la tienes, descargala aquí: 
[Versión 11.8 de Cuda](https://developer.nvidia.com/cuda-11-8-0-download-archive)

2. Instalar Deep Graph Library (DGL).
  Se puede instalar desde [aquí](https://www.dgl.ai/pages/start.html) o seguir los siguientes pasos en Linux:
```
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

3. Instalar las dependencias necesarias.
```
pip install -r requirements.txt
```

4. Puede haber la posibilidad de que no se haya instalado pytorch en la versión de cu118, para ello, ejecutamos el siguiente comando:
```
pip install torch==1.12.1+cu118 torchvision==0.13.2+cu118 torchaudio==0.12.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
``` 
## Estructura del repositorio
- data/ : Contiene el dataset de S-FFSD en diferentes archivos .cvs.
- methods/ : Contiene las implementaciones de los modelos, en este caso, solo el GTAN.
- models/ : Contiene el modelo GTAN preentrenado. Su extensión es .pth.

## Referencias
El código presente esta extraido del repositorio [Antifraud](https://github.com/finint/antifraud)
