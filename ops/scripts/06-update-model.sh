# This script simulates a model update by simply updating the last modified time of the local model tokenizer file.
# The expectation is that the `loader` module in the client app will detect the file change and push the "new" model to GemFire.
# GemFire will send an event upon the update to the AiModel rgion.
# The app will react to the event by pulling the new copy of the model.

echo "Simulating an update to the model tokenizer on the local filesystem"
echo "--> Model module in client app will detect the change ot the file and upload it to the GemFire AiModel Region"
echo "--> GemFire will fire an event and the app will be notified of the update."
echo "--> The app will pull the updated version of the model from GemFire."

echo " " >> models/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tokenizer.json

echo
echo
echo "Update complete! To verify that the app has pulled the new model, check the client app log file for the following logging: "
echo "=========="
echo "Detected change in AI assets (model: false, tokenizer: true). Refreshing GemFire entry."
echo "..."
echo "Updating local Onnx session and tokenizer with new model."
echo "=========="