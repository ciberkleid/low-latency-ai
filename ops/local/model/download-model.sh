# Model: distilbert-base-uncased-finetuned-sst-2-english
#
# A lightweight transformer model for binary sentiment analysis in English.
#
# - DistilBERT: a smaller, faster version of BERT created via knowledge distillation
# - Base: standard DistilBERT model size (6 transformer layers)
# - Uncased: input text is lowercased before processing
# - Fine-tuned on SST-2: trained specifically to classify sentiment
# - English-only: trained and evaluated on English text
#
# Note:
# This model outputs two logits: [negative, positive].
# It always predicts positive or negative (SST-2 has no neutral class).

DEST=models/distilbert/distilbert-base-uncased-finetuned-sst-2-english

rm -rf $DEST
mkdir -p $DEST
cd $DEST

wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx
wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/tokenizer.json

#wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/added_tokens.json
#wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/config.json
#wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/special_tokens_map.json
#wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/tokenizer_config.json
#wget https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/vocab.txt
