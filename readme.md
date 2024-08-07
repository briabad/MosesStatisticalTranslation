

# Proyecto de Traducción Automática

Este proyecto utiliza la distribución Moses para la traducción automática y ocupa los datasets de Europarl. A continuación, se detallan los pasos secuenciales para la tokenización, truecasing, limpieza, entrenamiento de modelos y evaluación del proceso de traducción.

## Requisitos Previos

- Docker
- Python
- fairseq
- SRILM

Asegurarse que todas las rutas y arhivos esten montados de manera correcta:

docker container run -it --rm -v ${PWD}/data/:/data moses /bin/bash 

## Descarga del Dataset de Europarl

Descarga los datasets de Europarl desde los siguientes enlaces oficiales:

- [Europarl v7 Spanish-English](http://www.statmt.org/europarl/v7/es-en.tgz)

Descomprime los archivos en el directorio `data/dataset`.

## 1. Tokenización de Datos

La tokenización es el proceso de dividir el texto en unidades más pequeñas, como palabras o subpalabras. Este paso es crucial porque los modelos de traducción automática trabajan mejor con unidades más pequeñas y consistentes.


Para espanol:
```sh
docker container run -it --rm -v ${PWD}/data/:/data moses /bin/bash -c "/opt/moses/scripts/tokenizer/tokenizer.perl -l es < /data/dataset/europarl-v7.es-en-train-red.es > /data/dataset/europarl-traincorpus.tok.es"

```
Para ingles: 

```sh
docker container run -it --rm -v ${PWD}/data/:/data moses /bin/bash -c "/opt/moses/scripts/tokenizer/tokenizer.perl -l en < /data/dataset/europarl-v7.es-en-train-red.en > /data/dataset/europarl-traincorpus.tok.en"

```



## 3. Limpieza de Datos

La limpieza de datos elimina oraciones demasiado largas o cortas, así como oraciones mal alineadas, mejorando la calidad del corpus.

Long sentences and empty sentences are removed as they can cause problems
with the training pipeline, and obviously mis-aligned sentences are removed.


```sh
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/clean-corpus-n.perl /data/dataset/europarl-traincorpus.tok es en /data/dataset/europarl-clean.tok 1 80
```

## 4. Lowercase

Convertir todo el texto a minúsculas para reducir la esparsidad de los datos.

```sh
docker container run -it --rm -v ${PWD}/:/data moses \
/opt/moses/scripts/tokenizer/lowercase.perl < /data/dataset/europarl-clean.tok.en > /data/dataset/europarl-clean-lower-tok.en
```

docker container run -it --rm -v ${PWD}/:/data moses \
/opt/moses/scripts/training/lowercase.perl < /data/europarl-traincorpus.tok.clean.en > /data/europarl-traincorpus.tok.clean.lowercase.en

## 5. Separación de Datasets

Dividir el corpus en conjuntos de entrenamiento y validación.

```sh
# Creación del modelo (80%)
head -n 48000 data/dataset/europarl-traincorpus.tok.clean.es > data/train-europarl48000.tok.clean.es
head -n 48000 data/dataset/europarl-traincorpus.tok.clean.en > data/train-europarl48000.tok.clean.en

# Ajuste de pesos del modelo (20%)
tail -n 2000 data/dataset/europarl-traincorpus.tok.clean.es > data/train-europarl2000.tok.clean.es
tail -n 2000 data/dataset/europarl-traincorpus.tok.clean.en > data/train-europarl2000.tok.clean.en
```

## 6. Creación de N-gramas

Construcción del modelo de lenguaje basado en n-gramas.

```sh
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/srilm/lm/bin/i686-m64/ngram-count -order 5 -unk -interpolate -kndiscount \
-text /data/train/train_bpe_corto.es -lm /data/model.lm
```

## 7. Entrenamiento del Modelo de Traducción

### Alignment Model

Entrenar el modelo de alineación utilizando GIZA++.

```sh
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/train-model.perl \
-root-dir /data/alignment \
-mgiza -mgiza-cpus 15 \
-corpus /data/train/train_bpe_corto -f en -e es \
-alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:5:/data/model.lm \
-external-bin-dir /opt/moses/mgiza/mgizapp/bin/
```

### Entrenamiento de los Pesos del Modelo Log-lineal (MERT)

Optimizar los pesos del modelo log-lineal usando Minimum Error Rate Training (MERT).

```sh
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/mert-moses.pl /data/train/training.clean.es \
/data/train/training.clean.en \
/opt/moses/bin/moses /data/alignment/model/moses.ini \
--maximum-iterations=8 \
--working-dir /data/mert \
--mertdir /opt/moses/bin/ \
--decoder-flags "-threads 15"
```

### Entrenamiento de los Pesos del Modelo Log-lineal (MERT con MIRA)

Optimización avanzada usando Hope-Fear MIRA.

```sh
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/mert-moses.pl /data/train/valid_bpe.en \
/data/train/development.clean.en \
/opt/moses/bin/moses /data/alignment/model/moses.ini \
--batch-mira --batch-mira-args "-J 300" \
--return-best-dev \
--maximum-iterations=5 \
--working-dir /data/mert \
--mertdir /opt/moses/bin/ \
--decoder-flags "-threads 10"
```

## 8. Proceso de Traducción

Utilizar el modelo entrenado para traducir texto.

```sh
docker container run -i --rm -v ${PWD}/data/:/data moses \
/opt/moses/bin/moses -threads 10 -f /data/mert/moses.proyecto.ini \
-i /data/test/test.es > data/test/testwb.hyp
```

## 9. Evaluación del Proceso de Traducción

Evaluar la calidad de la traducción utilizando la métrica BLEU.

```sh
docker container run -i --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/generic/multi-bleu.perl /data/test/test.en < data/test/testwb.hyp
```

## 10. Fairseq

### Preparación de los Datos

```sh
fairseq-preprocess --source-lang en --target-lang es \
    --trainpref europarl.tokenized.en-es/train --validpref europarl.tokenized.en-es/valid --testpref europarl.tokenized.en-es/test \
    --destdir data-bin/europarl.tokenized.en-es
```

### Entrenamiento con Fairseq

```sh
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/europarl.tokenized.en-es \
    --optimizer nag --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir checkpoints/fconv
```

### Generación con Fairseq

```sh
fairseq-generate data-bin/europarl.tokenized.en-es \
    --path checkpoints/fconv/checkpoint190.pt \
    --batch-size 128 --beam 5
```

---

Este README debería proporcionar una guía clara y completa para ejecutar el proyecto de traducción automática utilizando Moses y Fairseq, desde la descarga del dataset hasta la evaluación del modelo entrenado.