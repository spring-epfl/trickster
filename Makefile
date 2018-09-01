TEMP_DIR=.tmp

.PHONY: data
data: ${TEMP_DIR}/knndata.zip
	unzip ${TEMP_DIR}/knndata.zip -d data/
	mv data/batch data/knndata

.tmp/knndata.zip:
	mkdir -p ${TEMP_DIR}
	curl https://www.cse.ust.hk/~taow/wf/data/knndata.zip > ${TEMP_DIR}/knndata.zip

.PHONY: format
format:
	black scripts
	black trickster
	black tests

