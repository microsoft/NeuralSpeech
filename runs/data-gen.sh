EXP_HOME=$(cd `dirname $0`; pwd)/..
cd $EXP_HOME

cd FastCorrect
pip uninstall uuid
pip install --editable .
cd $EXP_HOME

DATA_PATH=<Path-to-Binary-Data>
export PYTHONPATH=$EXP_HOME/FastCorrect:$PYTHONPATH

TEXT=<Path-to-Data-with-duration>
# contains train.zh_CN, train.zh_CN_tgt, valid.zh_CN, valid.zh_CN_tgt
# *.zh_CN is indeed the *.src.werdur.full in alignment result
# *.zh_CH_tgt is indeed the *.tgt in alignment result


#We use shared dictionary extracted from training corpus

fairseq-preprocess --source-lang zh_CN --target-lang zh_CN_tgt \
    --task translation \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --padding-factor 8 \
    --src-with-werdur \
    --destdir ${DATA_PATH} \
    --srcdict '<path-to-dictionary>' --tgtdict '<path-to-dictionary>' \
    --workers 20
