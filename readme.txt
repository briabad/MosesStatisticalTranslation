/mnt/c/Users/brian/OneDrive/Escritorio/PROJECTS/UPV/traduccion_automatica_moses/src


*********************************************************************************************************
tokenizacion de datos
*********************************************************************************************************
Si no funciona correr dentro del contenedor

docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/tokenizer/tokenizer.perl -l en < /data/dataset/europarl-v7.es-en-train-red.es > /data/dataset/europarl-traincorpus.tok.es


*********************************************************************************************************
truecasing: 
*********************************************************************************************************
The initial words in each sentence are converted to their most probable casing. This helps reduce data sparsity.

docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/recaser/train-truecaser.perl \
--model /data/model/true_case_model_europarl.en --corpus /data/dataset/europarl-traincorpus.tok.en


docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/recaser/train-truecaser.perl \
--model /data/model/true_case_model_europarl.en --corpus /data/dataset/europarl-traincorpus.tok.en

*********************************************************************************************************
limpieza de datos
*********************************************************************************************************
Si no funciona correr dentro del contenedor

docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/clean-corpus-n.perl /data/train/train_bpe es en /data/train/train_bpe_corto 1 80

docker container run -it --rm -v ${PWD}/:/data moses \
/opt/moses/scripts/training/clean-corpus-n.perl /data/europarl-v7.es-en-test.tok es en /data/europarl-v7.es-en-test.tok.clean 1 1000


/opt/moses/scripts/training/clean-corpus-n.perl /data/europarl-traincorpus.tok es en /data/europarl-traincorpus.tok.clean 1 80

*********************************************************************************************************
lowercase
*********************************************************************************************************
docker container run -it --rm -v ${PWD}/:/data moses \
/opt/moses/scripts/training/lowercase.perl < /data/europarl-traincorpus.tok.clean.en > /data/europarl-traincorpus.tok.clean.lowercase.en

*********************************************************************************************************
Separaciono de datasets
*********************************************************************************************************
# Creación del modelo (80%)
head -n 48000 europarl-traincorpus.tok.clean.es > train-europarl48000.tok.clean.es
head -n 48000 europarl-traincorpus.tok.clean.en > train-europarl48000.tok.clean.en


# Ajuste de pesos del modelo (20%)
tail -n 2000 europarl-traincorpus.tok.clean.es > train-europarl2000.tok.clean.es
tail -n 2000 europarl-traincorpus.tok.clean.en > train-europarl2000.tok.clean.en

*********************************************************************************************************
creacion ngrama
*********************************************************************************************************

docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/srilm/lm/bin/i686-m64/ngram-count -order 5 -unk -interpolate -kndiscount  \
-text /data/train/train_bpe_corto.es -lm /data/model.lm

ngram-count -order 3 -unk -interpolate \
-kndiscount -text training.clean.en -lm lm/turista.lm


*********************************************************************************************************
Entrenamiento del modelo de traducci´on, entrenamiento ngrama
Alignment model
*********************************************************************************************************

docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/train-model.perl \
-root-dir /data/alignment  \
-mgiza -mgiza-cpus 15 \
-corpus /data/train/train_bpe_corto -f en -e es \
-alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:5:/data/model.lm \
-external-bin-dir /opt/moses/mgiza/mgizapp/bin/




*********************************************************************************************************
Entrenamiento de los pesos del modelo log-lineal
MERT
*********************************************************************************************************

docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/mert-moses.pl /data/train/training.clean.es \
/data/train/training.clean.en \
/opt/moses/bin/moses /data/alignment/model/moses.ini \
--maximum-iterations=8 \
--working-dir /data/mert \
--mertdir /opt/moses/bin/ \
--decoder-flags "-threads 15" \



*********************************************************************************************************
Entrenamiento de los pesos del modelo log-lineal
MERT CON MIRA
*********************************************************************************************************
This is hope-fear MIRA built as a drop-in replacement for MERT; it conducts online training
using aggregated k-best lists as an approximation to the decoder’s true search space. This
allows it to handle large features, and it often out-performs MERT once feature counts get
above 10.
You can tune using this system by adding --batch-mira to your mert-moses.pl command. This
replaces the normal call to the mert executable with a call to kbmira.
I recommend also adding the flag --return-best-dev to mert-moses.pl. This will copy the
moses.ini file corresponding to the highest-scoring development run (as determined by the
evaluator executable using BLEU on run*.out) into the final moses.ini. This can make a fairly
big difference for MIRA’s test-time accuracy

You can also pass through options to kbmira by adding --batch-mira-args ’whatever’ to
mert-moses.pl. Useful kbmira options include:
• -J n : changes the number of inner MIRA loops to n passes over the data. Increasing this
value to 100 or 300 can be good for working with small development sets. The default,
60, is ideal for development sets with more than 1000 sentences.
• -C n : changes MIRA’s C-value to n. This controls regularization. The default, 0.01,
works well for most situations, but if it looks like MIRA is over-fitting or not converging,
decreasing C to 0.001 or 0.0001 can sometimes help.
• --streaming : stream k-best lists from disk rather than load them into memory. This
results in very slow training, but may be necessary in low-memory environments or with
very large development sets.
Run kbmira --help for a full list of options.


docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/mert-moses.pl /data/train/valid_bpe.en \
/data/train/development.clean.en \
/opt/moses/bin/moses /data/alignment/model/moses.ini \
--batch-mira --batch-mira-args "-J 300" \
--return-best-dev \
--maximum-iterations=5 \
--working-dir /data/mert \
--mertdir /opt/moses/bin/ \
--decoder-flags "-threads 10" \s



 mv /mnt/c/Users/brian/OneDrive/Escritorio/PROJECTS/UPV/traduccion_automatica/src/data/mert/moses.ini /mnt/c/Users/brian/OneDrive/Escritorio/PROJECTS/UPV/traduccion_automatica/src/data/mert/moses.proyecto.ini

*********************************************************************************************************
PROCESO DE TRADUCCION
*********************************************************************************************************

docker container run -i --rm -v ${PWD}/data/:/data moses \
bin/moses -threads 10 -f /data/mert/moses.proyecto.ini \
-i /data/test/test.es > data/test/testwb.hyp

*********************************************************************************************************
Evaluaci´on del proceso de traducci´on
*********************************************************************************************************

docker container run -i --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/generic/multi-bleu.perl /data/test/test.en < data/test/testwb.hyp

*********************************************************************************************************
fairseq
*********************************************************************************************************
preparacion de los datos

fairseq-preprocess --source-lang en --target-lang es \
    --trainpref europarl.tokenized.en-es/train --validpref europarl.tokenized.en-es/valid --testpref europarl.tokenized.en-es/test \
    --destdir data-bin/europarl.tokenized.en-es

Entrenamienot fairseq

CUDA_VISIBLE_DEVICES=0 fairseq-train  data-bin/europarl.tokenized.en-es \
    --optimizer nag --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir checkpoints/fconv


fairseq-generate data-bin/europarl.tokenized.en-es \
    --path checkpoints/fconv/checkpoint190.pt \
    --batch-size 128 --beam 5

C:\Users\brian\OneDrive\Escritorio\PROJECTS\UPV\traduccion_automatica_opennmt\fairseq\examples\translation\europarl.tokenized.en-es\train.en
C:\Users\brian\OneDrive\Escritorio\PROJECTS\UPV\traduccion_automatica_opennmt\fairseq\examples\translation\checkpoints\fconv\checkpoint190.pt