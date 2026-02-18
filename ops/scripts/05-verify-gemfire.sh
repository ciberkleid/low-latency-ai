#!/usr/bin/env bash
set -euo pipefail

### Connect using gfsh:
# docker exec -it gf-locator gfsh
# connect

### At gfsh prompt, simply run commands:
# list regions
# list functions
# show metrics
# show metrics --region=/SentimentResults
# describe region --name=SentimentResults
# show log --member=server1 --lines=100

### After starting GemFire and Spring app, verify (1) AiModel region (2) initial cached sentiment result,
###   and (3) loaded product reviews data (ProductReviews region)
###   (1) Should show single entry with key="sentiment":
# query --query="select * from /AiModel.keys"
###   (2) Should show entry with key="Woohoo! ... Well done!" and value="POSITIVE":
# query --query="select key,value from /SentimentResults.entries"
###   (3) Should show all data in ProductReviews region
# query --query="select key,value.productName,value.review from /ProductReviews.entries"

### After sending edge requests, verify cached sentiment results
# query --query="select key,value from /SentimentResults.entries"

### After sending function requests, verify cached sentiment results
### First get unique sentiments for a given product:
# query --query="select value.review, count(*) as reviewCount from /ProductReviews.entries where value.productName='Lawnmower' group by value.review"
# query --query="select value.review, count(*) as reviewCount from /ProductReviews.entries where value.productName='Coffee Mug' group by value.review"
### Then check cached sentiment results:
# query --query="select key,value.productName,value.review from /ProductReviews.entries"

### Any of these commands can also be run at a regular terminal by executing them in the Docker locator container.
# For example:

# docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "list regions"
# docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "list functions"
docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "query --query='select key,value from /SentimentResults.entries'"

### Note that you can also invoke the function from gfsh:
# execute function --id=countPositiveReviews --region=/ProductReviews --arguments=Lawnmower

### Miscellaneous comment -- the following syntax works without PDX:
### query --query="select key,value from /ProductReviews.entries"
### query --query="select key,value.productName,value.review from /ProductReviews.entries where value.productName='Lawnmower'"
