
# Modelo de Alineación en Traducción Automática con Moses

En la traducción automática, especialmente utilizando Moses, el modelo de alineación es crucial para mapear palabras entre dos lenguas. A continuación se presenta un detalle matemático de cómo funciona un modelo de alineación y por qué es necesario.

## Modelo del Lenguaje
Un modelo de lenguaje basado en n-gramas es una técnica fundamental en procesamiento de lenguaje natural (NLP) que se utiliza para predecir la probabilidad de una secuencia de palabras. A continuación se presenta un detalle estadístico y matemático de cómo funcionan estos modelos.

#### Definición de N-grama

Un n-grama es una subsecuencia contigua de n elementos (palabras) de una secuencia dada. Por ejemplo:

- **Unigrama (1-grama)**: "yo", "amo", "programar"
- **Bigramas (2-gramas)**: "yo amo", "amo programar"
- **Trigramas (3-gramas)**: "yo amo programar"

La probabilidad de una secuencia de palabras usando un modelo de n-gramas se define como:

\[ P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1, w_2) \cdot ... \cdot P(w_n | w_{n-(n-1)}, ..., w_{n-1}) \]

Donde \( P(w_i | w_{i-(n-1)}, \ldots, w_{i-1}) \) es la probabilidad de la palabra \( w_i \) dada las \( n-1 \) palabras anteriores.

#### Probabilidad de N-gramas

El objetivo de un modelo de n-gramas es estimar la probabilidad de una palabra dada su historial de n-1 palabras anteriores. La fórmula general para un modelo de n-gramas es:

\[ P(w_i | w_{i-(n-1)}, \ldots, w_{i-1}) \]

Donde \( w_i \) es la palabra actual y \( w_{i-(n-1)}, \ldots, w_{i-1} \) son las n-1 palabras anteriores.

#### Ejemplo de Trigrama

Para un trigramo (n=3), la probabilidad de la palabra \( w_i \) dada las dos palabras anteriores \( w_{i-2} \) y \( w_{i-1} \) es:

\[ P(w_i | w_{i-2}, w_{i-1}) \]

#### Estimación de Probabilidades

Las probabilidades de n-gramas se estiman a partir de frecuencias observadas en un corpus de entrenamiento. La probabilidad de un n-grama se calcula como:

\[ P(w_i | w_{i-(n-1)}, \ldots, w_{i-1}) = \frac{C(w_{i-(n-1)}, \ldots, w_{i-1}, w_i)}{C(w_{i-(n-1)}, \ldots, w_{i-1})} \]

Donde:
- \( C(w_{i-(n-1)}, \ldots, w_{i-1}, w_i) \) es la frecuencia del n-grama en el corpus.
- \( C(w_{i-(n-1)}, \ldots, w_{i-1}) \) es la frecuencia del (n-1)-grama en el corpus.

### Ejemplo de Bigramas

Para un bigrama (n=2), la probabilidad se calcula como:

\[ P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})} \]


La perplejidad es una métrica común para evaluar la calidad de un modelo de lenguaje. Se define como la exponencial de la entropía cruzada de la secuencia:

\[ PP(W) = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_{i-(n-1)}, \ldots, w_{i-1})} \]

Donde \( N \) es el número total de palabras en la secuencia.

Una perplejidad más baja indica un mejor rendimiento del modelo de lenguaje.




##  Modelo de Alineación

En GIZA++, el modelo de lenguaje se utiliza junto con el modelo de alineación para evaluar y seleccionar las mejores traducciones posibles. Primero, el modelo de alineación (por ejemplo, Modelo IBM 1) se entrena para mapear palabras del idioma de origen al idioma de destino, calculando las probabilidades de traducción para cada par de palabras. Luego, se entrena un modelo de lenguaje en el idioma de destino para evaluar la probabilidad de las secuencias de palabras, utilizando técnicas como n-gramas o redes neuronales. Las probabilidades de alineación se combinan con las probabilidades del modelo de lenguaje, y la probabilidad total de una secuencia traducida se calcula como el producto de ambas probabilidades. Durante la decodificación, se selecciona la traducción que maximiza esta probabilidad total combinada, asegurando que la traducción sea correcta en términos de alineación, fluida y gramaticalmente correcta en el idioma de destino.

El modelo de alineación en la traducción automática tiene como objetivo principal mapear palabras y frases entre un idioma de origen y un idioma de destino, considerando que las estructuras de las frases y las palabras correspondientes pueden variar significativamente entre los idiomas. Esto es esencial para traducir correctamente y facilitar la creación de una tabla de frases de alta calidad, que es fundamental para los sistemas de traducción basados en frases como Moses. Además, el modelo de alineación ayuda a identificar dependencias lingüísticas, manejando fenómenos como la expansión o contracción de frases, y mejora significativamente la precisión de la traducción al alinear correctamente palabras y frases entre dos idiomas, produciendo así traducciones coherentes y contextualmente apropiadas.

## Cómo Funciona un Modelo de Alineación

### Modelo Generativo

Supongamos que tenemos una oración en el idioma de origen \( f = (f_1, f_2, ..., f_m) \) y una oración en el idioma de destino \( e = (e_1, e_2, ..., e_l) \). El objetivo del modelo de alineación es estimar la probabilidad de que una palabra en el idioma de origen \( f_i \) se alinee con una palabra en el idioma de destino \( e_j \).

### Modelo IBM 1

El Modelo IBM 1 es el modelo de alineación más simple y asume que todas las palabras en el idioma de destino son generadas independientemente por las palabras en el idioma de origen. La probabilidad de una oración en el idioma de destino dada una oración en el idioma de origen se define como:

\[ P(e | f) = \prod_{i=1}^{l} \sum_{j=0}^{m} t(e_i | f_j) \]

Donde:
- \( t(e_i | f_j) \) es la probabilidad de traducir la palabra \( f_j \) en la palabra \( e_i \).
- \( f_0 \) es una palabra especial de NULL que permite manejar inserciones en el idioma de destino.

### Modelo IBM 2

El Modelo IBM 2 introduce el concepto de posiciones y considera la probabilidad de que una palabra en una posición en el idioma de origen se alinee con una palabra en una posición en el idioma de destino. La probabilidad de alineación se define como:

\[ P(e, a | f) = \prod_{i=1}^{l} t(e_i | f_{a_i}) \cdot d(a_i | i, l, m) \]

Donde:
- \( a \) es una alineación, una secuencia de índices que indican qué palabra en el idioma de origen genera cada palabra en el idioma de destino.
- \( d(a_i | i, l, m) \) es la probabilidad de que la posición \( i \) en la oración del idioma de destino se alinee con la posición \( a_i \) en la oración del idioma de origen.

### Entrenamiento de Modelos de Alineación

El entrenamiento de los modelos de alineación se realiza generalmente utilizando el algoritmo EM (Expectation-Maximization). A continuación, se describen brevemente las etapas del algoritmo EM en el contexto de los modelos de alineación.

#### Etapa de Expectativa (E-Step)

En esta etapa, se calcula la expectativa de las alineaciones dadas las probabilidades actuales de traducción y alineación. Esto implica calcular las probabilidades posteriores de las posibles alineaciones:

\[ P(a_i = j | e, f) = \frac{t(e_i | f_j) \cdot d(j | i, l, m)}{\sum_{k=0}^{m} t(e_i | f_k) \cdot d(k | i, l, m)} \]

#### Etapa de Maximización (M-Step)

En esta etapa, se actualizan las probabilidades de traducción y alineación para maximizar la probabilidad de los datos observados. Las actualizaciones se realizan utilizando las expectativas calculadas en la etapa de expectativa:

\[ t(e_i | f_j) = \frac{\sum_{(e, f)} C(e_i, f_j)}{\sum_{(e, f)} \sum_{k=1}^{l} C(e_k, f_j)} \]

\[ d(j | i, l, m) = \frac{\sum_{(e, f)} C(a_i = j | e, f)}{\sum_{(e, f)} \sum_{k=1}^{m} C(a_i = k | e, f)} \]

Donde \( C(e_i, f_j) \) es el conteo esperado de la palabra \( e_i \) siendo alineada con la palabra \( f_j \), y \( C(a_i = j | e, f) \) es el conteo esperado de la posición \( i \) en la oración de destino alineada con la posición \( j \) en la oración de origen.

### Visualización de Alineaciones

Las alineaciones aprendidas por los modelos pueden visualizarse como matrices de alineación, donde las filas corresponden a palabras en el idioma de destino y las columnas a palabras en el idioma de origen. Cada celda en la matriz representa la probabilidad de que una palabra en una posición en el idioma de origen se alinee con una palabra en una posición en el idioma de destino.


## Modelo Log-Lineal

Un modelo log-lineal MERT (Minimum Error Rate Training) es una técnica utilizada en la traducción automática para optimizar los parámetros de un modelo log-lineal con el objetivo de minimizar la tasa de error en las traducciones generadas. A continuación se describe en detalle qué es un modelo log-lineal y cómo se aplica MERT en este contexto.

Un modelo log-lineal es una técnica que permite combinar múltiples características o factores que influyen en la selección de la mejor traducción en un marco unificado. En el contexto de la traducción automática, estas características pueden incluir probabilidades de alineación, probabilidades del modelo de lenguaje, y otros factores como la longitud de la secuencia. La puntuación de una traducción \( e \) dada una oración en el idioma de origen \( f \) se calcula como una combinación ponderada de características:

\[ h(e, f) = \sum_{i} \lambda_i \cdot h_i(e, f) \]

Donde \( h_i(e, f) \) son las características individuales (por ejemplo, probabilidades de traducción, probabilidades del modelo de lenguaje, etc.) y \( \lambda_i \) son los pesos que deben ser ajustados.

## Minimum Error Rate Training (MERT)

### Definición

Minimum Error Rate Training (MERT) es una técnica de optimización utilizada para ajustar los pesos de un modelo log-lineal en la traducción automática. El objetivo de MERT es minimizar la tasa de error en las traducciones generadas, típicamente medida mediante una métrica de evaluación como BLEU (Bilingual Evaluation Understudy).

### Funcionamiento de MERT

1. **Entrenamiento del Modelo de Alineación**: El modelo de alineación (por ejemplo, Modelo IBM 1) se entrena para mapear palabras del idioma de origen al idioma de destino, calculando las probabilidades de traducción para cada par de palabras.

2. **Evaluación del Modelo de Lenguaje**: Se entrena un modelo de lenguaje en el idioma de destino para evaluar la probabilidad de las secuencias de palabras, utilizando técnicas como n-gramas o redes neuronales.

3. **Combinar Probabilidades**: Las probabilidades de alineación se combinan con las probabilidades del modelo de lenguaje. La probabilidad total de una secuencia traducida se calcula como el producto de ambas probabilidades.

4. **Decodificación**: Durante la decodificación, se selecciona la traducción que maximiza esta probabilidad total combinada. Esto asegura que la traducción sea correcta en términos de alineación, fluida y gramaticalmente correcta en el idioma de destino.


### Aplicación en Moses

En Moses, MERT se implementa como un proceso de ajuste posterior al entrenamiento inicial del modelo de alineación y traducción. Aquí se muestra cómo se ejecuta MERT en Moses:

1. **Preparación del Conjunto de Datos de Desarrollo**: Se selecciona un conjunto de datos de desarrollo que contiene pares de oraciones en el idioma de origen y su correspondiente traducción de referencia.

2. **Ejecución de MERT**:
   ```sh
   # Ejecutar MERT en Moses
   moses/scripts/training/mert-moses.pl \
       /path/to/development/source.txt \
       /path/to/development/reference.txt \
       moses/bin/moses \
       /path/to/model/moses.ini \
       --working-dir /path/to/mert-working-dir \
       --decoder-flags "-threads 8"
   ```

3. **Optimización Iterativa**: MERT ajusta iterativamente los pesos de las características para minimizar el error en las traducciones del conjunto de datos de desarrollo.


# Evaluacion (BLEU)

BLEU (Bilingual Evaluation Understudy) es una métrica de evaluación utilizada para medir la calidad de la traducción automática comparando una traducción generada por máquina con una o más traducciones de referencia realizadas por humanos. Matemáticamente, BLEU calcula la precisión de n-gramas entre la traducción automática y las traducciones de referencia. A continuación se explica el funcionamiento matemático de BLEU en detalle:

## Cálculo de BLEU

### 1. Precisión de N-gramas

La precisión de n-gramas mide la proporción de n-gramas en la traducción generada por la máquina que también aparecen en las traducciones de referencia. Se calculan precisiones separadas para diferentes valores de n (generalmente n = 1, 2, 3, 4).

Para calcular la precisión de n-gramas:

\[ \text{Precisión de n-gramas} = \frac{\sum_{\text{todos los n-gramas}} \min(\text{conteo de n-gramas en la traducción generada}, \text{conteo de n-gramas en las referencias})}{\sum_{\text{todos los n-gramas}} \text{conteo de n-gramas en la traducción generada}} \]

### 2. Penalización por Brevedad (Brevity Penalty, BP)

Para evitar que las traducciones cortas obtengan puntuaciones artificialmente altas, BLEU incluye una penalización por brevedad. La penalización por brevedad se calcula como:

\[ BP = 
\begin{cases} 
1 & \text{si } c > r \\
e^{(1 - \frac{r}{c})} & \text{si } c \leq r 
\end{cases}
\]

Donde:
- \( c \) es la longitud total de la traducción generada.
- \( r \) es la longitud total de las traducciones de referencia más cercanas en longitud a la traducción generada.

### 3. Cálculo de la Puntuación BLEU

La puntuación BLEU se calcula combinando las precisiones de n-gramas y la penalización por brevedad. Primero, se calcula la media geométrica de las precisiones de n-gramas:

\[ \text{Precisión de n-gramas combinada} = \exp \left( \frac{1}{N} \sum_{n=1}^{N} \log p_n \right) \]

Donde:
- \( N \) es el número máximo de n-gramas considerados (generalmente 4).
- \( p_n \) es la precisión de n-gramas para el valor \( n \).

Finalmente, la puntuación BLEU se obtiene multiplicando la precisión de n-gramas combinada por la penalización por brevedad:

\[ \text{BLEU} = BP \times \exp \left( \frac{1}{N} \sum_{n=1}^{N} \log p_n \right) \]

### Ejemplo de Cálculo de BLEU

Supongamos que tenemos una traducción generada por la máquina y una traducción de referencia. Para simplificar, consideremos solo unigramas (n=1).

- Traducción generada: "el gato está sobre la alfombra"
- Traducción de referencia: "el gato está en la alfombra"

#### 1. Cálculo de la Precisión de Unigramas

- Unigramas en la traducción generada: "el", "gato", "está", "sobre", "la", "alfombra"
- Unigramas en la traducción de referencia: "el", "gato", "está", "en", "la", "alfombra"

Precisión de unigramas:

\[ p_1 = \frac{4}{6} = 0.6667 \]

(4 unigramas coinciden: "el", "gato", "está", "la")

#### 2. Cálculo de la Penalización por Brevedad

- Longitud de la traducción generada (\( c \)): 6
- Longitud de la traducción de referencia (\( r \)): 6

Penalización por brevedad:

\[ BP = 1 \]

#### 3. Cálculo de la Puntuación BLEU

\[ \text{BLEU} = BP \times \exp \left( \frac{1}{1} \log p_1 \right) = 1 \times 0.6667 = 0.6667 \]

Por lo tanto, la puntuación BLEU para este ejemplo simplificado sería 0.6667.


## Referencias

1. Brown, P. F., Della Pietra, S. A., Della Pietra, V. J., & Mercer, R. L. (1993). "The mathematics of statistical machine translation: Parameter estimation." *Computational Linguistics, 19*(2), 263-311.
   - Este artículo seminal presenta los Modelos IBM 1 a 5 y describe en detalle los fundamentos matemáticos de la alineación y la traducción estadística.

2. Och, F. J., & Ney, H. (2003). "A Systematic Comparison of Various Statistical Alignment Models." *Computational Linguistics, 29*(1), 19-51.
   - Este trabajo compara varios modelos de alineación estadística y describe el uso del algoritmo EM para el entrenamiento de modelos de alineación.

3. Koehn, P. (2010). *Statistical Machine Translation*. Cambridge University Press.
   - Este libro ofrece una visión completa de la traducción automática estadística, incluyendo una explicación detallada de los modelos de alineación y su implementación en herramientas como Moses.

4. GIZA++ Software. 
   - La implementación de los modelos de alineación descritos en los Modelos IBM está disponible en el software GIZA++. La documentación y el código fuente proporcionan detalles prácticos sobre la implementación y el uso de estos modelos.
   - [GIZA++ Documentation](http://www.statmt.org/moses/giza/GIZA++.html)


1. **Och, F. J. (2003).** "Minimum Error Rate Training in Statistical Machine Translation." *Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics*, 160-167.
   - Este artículo describe el método de MERT y su aplicación en la traducción automática estadística.

2. **Koehn, P. (2010).** *Statistical Machine Translation*. Cambridge University Press.
   - Proporciona una visión detallada de MERT y otros métodos de optimización utilizados en la traducción automática estadística.

3. **Moses Documentation**:
   - [Moses Documentation on MERT](http://www.statmt.org/moses/?n=FactoredTraining.MERT)
   - Documentación sobre la implementación y uso de MERT en Moses.

1. **Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002).** "BLEU: a method for automatic evaluation of machine translation." *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*, 311-318.
   - Este artículo presenta la métrica BLEU y describe su cálculo en detalle.



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
/opt/moses/scripts/training/clean-corpus-n.perl /data/dataset/europarl-traincorpus.tok es en /data/dataset/europarl-clean-tok 1 80
```

## 4. Lowercase

Convertir todo el texto a minúsculas para reducir la esparsidad de los datos.

Para ingles
```sh
docker container run -it --rm -v ${PWD}/data/:/data moses  /bin/bash -c " /opt/moses/scripts/tokenizer/lowercase.perl < /data/dataset/europarl-clean-tok.en > /data/dataset/europarl-clean-lower-tok.en"
```
Para espanol
```sh
docker container run -it --rm -v ${PWD}/data/:/data moses  /bin/bash -c " /opt/moses/scripts/tokenizer/lowercase.perl < /data/dataset/europarl-clean-tok.es > /data/dataset/europarl-clean-lower-tok.es"
```


## 5. Separación de Datasets
Dividir el corpus en conjuntos de entrenamiento y validación.

```sh
# Creación del modelo (80%)
head -n 48000 ./data/dataset/europarl-clean-lower-tok.es > data/train/train-europarl48000.tok.clean.es
head -n 48000 ./data/dataset/europarl-clean-lower-tok.en > data/train/train-europarl48000.tok.clean.en

# Ajuste de pesos del modelo (20%)
head -n 2000 ./data/dataset/europarl-clean-lower-tok.es > data/train/train-europarl2000.tok.clean.es
head -n 2000 ./data/dataset/europarl-clean-lower-tok.en > data/train/train-europarl2000.tok.clean.en
```

## 6. Creación de N-gramas


En el contexto de SRILM creamos un modelo de lenguaje con el siguiente codigo:

```sh
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/srilm/lm/bin/i686-m64/ngram-count -order 5 -unk -interpolate -kndiscount \
-text /data/train/train-europarl48000.tok.clean.es -lm /data/model.lm
```

## 7. Entrenamiento del Modelo de Traducción

### Alignment Model

Entrenar el modelo de alineación utilizando GIZA++. Recuerda configurar el parametro -lm de acuerdo a la configuracion de tu ngrama previamente entrenado

```sh
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/train-model.perl \
-root-dir /data/alignment \
-mgiza -mgiza-cpus 15 \
-corpus /data/train/train-europarl48000.tok.clean -f en -e es \
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