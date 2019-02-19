TEMP_DIR=.tmp

.PHONY: data
data: knndata twitter_bots

.PHONY: knndata
knndata: ${TEMP_DIR}/knndata.zip
	mkdir data/wfp
	unzip ${TEMP_DIR}/knndata.zip -d data/wfp

${TEMP_DIR}/knndata.zip:
	mkdir -p ${TEMP_DIR}
	curl https://www.cse.ust.hk/~taow/wf/data/knndata.zip > ${TEMP_DIR}/knndata.zip

.PHONY: twitter_bots
twitter_bots: ${TEMP_DIR}/twitter_bots.zip
	unzip ${TEMP_DIR}/twitter_bots.zip -d data/
	rm -rf data/twitter_bots
	mv data/classification_processed data/twitter_bots

${TEMP_DIR}/twitter_bots.zip:
	mkdir -p ${TEMP_DIR}
	curl https://www.cl.cam.ac.uk/~szuhg2/data/classification_processed.zip > ${TEMP_DIR}/twitter_bots.zip

.PHONY: format
format:
	black scripts
	black trickster
	black tests
