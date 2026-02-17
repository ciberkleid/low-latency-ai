# docker exec -it gf-locator gfsh
# connect

# show log --member=server1 --lines=100
# list regions
# list functions

# show metrics

# query --query="select * from /AiModel.keys"

# query --query="select key,value.productName,value.review from /ProductReviews.entries"
# query --query="select value.review, count(*) as reviewCount from /ProductReviews.entries where value.productName='Lawnmower' group by value.review"

# describe region --name=SentimentResults
# show metrics --region=/SentimentResults
# query --query="select key,value from /SentimentResults.entries"

docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "list functions"
docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "list regions"
docker exec -it gf-locator gfsh -e "connect --jmx-manager=gf-locator[1099]" -e "query --query='select key,value from /SentimentResults.entries'"


### Works without PDX:
### query --query="select key,value from /ProductReviews.entries"
### query --query="select key,value.productName,value.review from /ProductReviews.entries where value.productName='Lawnmower'"