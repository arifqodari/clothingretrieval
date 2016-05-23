.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
VER = v5
UNARY = unary_$(VER)
CRF = crf_$(VER)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

requirements:
	pip install -q -r requirements.txt

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --exclude=lib/,bin/ .

sync_data_to_s3:
	s3cmd sync --recursive data/ s3://$(BUCKET)/data/

sync_data_from_s3:
	s3cmd sync --recursive s3://$(BUCKET)/data/ data/

interim_clean:
	rm -rf data/interim/*

download_images:
	mkdir -p data/raw/$(group)
	mkdir -p data/raw/$(group)/images
	mkdir -p data/raw/$(group)/images/$(cat)
	python src/download_images.py --csv data/raw/$(group)/csv/query_result_$(cat).csv --out-dir data/raw/$(group)/images/$(cat)

od_train:
	# python src/od_train.py --anno-dir data/raw/ccp/annotations/csv --photo-dir data/raw/ccp/photos/ --model-filename data/interim/od/detector.pkl
	# python src/od_train.py --anno-dirs data/raw/ccp/annotations/csv data/raw/cfpd/annotations/csv --photo-dirs data/raw/ccp/photos/ data/raw/cfpd/images --model-filename data/interim/od/detector.pkl
	# python src/od_train.py --anno-dirs data/raw/ccp/annotations/csv --photo-dirs data/raw/ccp/photos/ --model-filename data/interim/od/detector.pkl
	python src/od_train.py --anno-dirs data/raw/cfpd/annotations/csv --photo-dirs data/raw/cfpd/images --model-filename data/interim/od/detector.pkl

od_detect:
	python src/od_detect.py --model-filename data/interim/od/detector.pkl --image-path $(image)

ccp_crop_person:
	mkdir -p data/raw/ccp/cropped_photos
	mkdir -p data/raw/ccp/annotations/cropped_csv
	python src/crop_person.py --anno-dir data/raw/ccp/annotations/csv --photo-dir data/raw/ccp/photos/ --output-photo-dir data/raw/ccp/cropped_photos/ --output-anno-dir data/raw/ccp/annotations/cropped_csv

ccp_remove_crop_person:
	rm -r data/raw/ccp/cropped_photos
	rm -r data/raw/ccp/annotations/cropped_csv

cfpd_remove_anotations:
	rm -r data/raw/cfpd/annotations/csv

cfpd_get_annotations:
	mkdir -p data/raw/cfpd/annotations/csv
	python src/cfpd_get_annotations.py --inp-dir data/raw/cfpd/annotations/images --out-dir data/raw/cfpd/annotations/csv

cfpd_crop_person:
	mkdir -p data/raw/cfpd/cropped_images
	mkdir -p data/raw/cfpd/annotations/cropped_csv
	python src/crop_person.py --anno-dir data/raw/cfpd/annotations/csv --photo-dir data/raw/cfpd/images/ --output-photo-dir data/raw/cfpd/cropped_images/ --output-anno-dir data/raw/cfpd/annotations/cropped_csv --max-size 500,200

cfpd_remove_crop_person:
	rm -r data/raw/cfpd/cropped_images
	rm -r data/raw/cfpd/annotations/cropped_csv

ss_crop_images:
	mkdir -p data/interim/$(cat)
	# mkdir -p data/raw/ss/cropped_images
	# mkdir -p data/raw/ss/cropped_images/$(cat)
	python src/ss_crop_images.py --inp-dir data/raw/ss/images/$(cat) --out-dir data/interim/$(cat) --cat $(cat)

ss_remove_cropped_images:
	rm -r data/raw/ss/cropped_images/$(cat)

ss_feature_extraction:
	mkdir -p data/processed/sim/ss
	mkdir -p data/processed/sim/ss/$(cat)
	python src/ss_feature_extraction.py --unary-model-path models/$(UNARY)/unary_clf.pkl --crf-model-path models/$(CRF)/crf_clf.pkl --cat $(cat) --inp-dir data/raw/ss/cropped_images/$(cat)/ --out-dir data/processed/sim/ss/$(cat)/

ss_retrieval:
	python src/ss_retrieval.py --unary-model-path models/$(UNARY)/unary_clf.pkl --crf-model-path models/$(CRF)/crf_clf.pkl --cat $(cat) --features data/processed/sim/ss/$(cat)/sim_features_$(cat).pkl --image-path $(image)

unary_feature_extraction:
	python src/unary_feature_extraction.py --anno-dir data/raw/ccp/annotations/csv_parts/$(part)/ --photo-dir data/raw/ccp/photos/ --output-dir data/interim/

unary_feature_extraction_cropped:
	python src/unary_feature_extraction.py --anno-dir data/raw/ccp/annotations/cropped_csv_parts/$(part)/ --photo-dir data/raw/ccp/cropped_photos/ --output-dir data/interim/

unary_concat_features:
	mkdir -p data/processed/unary
	mkdir -p data/processed/unary/$(UNARY)
	python src/concat_npy.py --out data/processed/unary/$(UNARY)/X_unary.npy --inp data/interim/X_unary_part*
	python src/concat_npy.py --out data/processed/unary/$(UNARY)/y_unary.npy --inp data/interim/y_unary_part*
	rm data/interim/X_unary_part*
	rm data/interim/y_unary_part*

unary_train:
	mkdir -p models/$(UNARY)
	python src/unary_train.py --data-file data/processed/unary/$(UNARY)/X_unary.npy --target-file data/processed/unary/$(UNARY)/y_unary.npy --model-filename models/$(UNARY)/unary_clf.pkl

unary_segmentation:
	python src/unary_segmentation.py --model-path models/$(UNARY)/unary_clf.pkl --image-path $(image)

crf_feature_extraction:
	python src/crf_feature_extraction.py --anno-dir data/raw/ccp/annotations/csv_parts/$(part)/ --photo-dir data/raw/ccp/photos/ --output-dir data/interim/ --unary-model models/$(UNARY)/unary_clf.pkl

crf_feature_extraction_cropped:
	python src/crf_feature_extraction.py --anno-dir data/raw/ccp/annotations/cropped_csv_parts/$(part)/ --photo-dir data/raw/ccp/cropped_photos/ --output-dir data/interim/ --unary-model models/$(UNARY)/unary_clf.pkl

crf_concat_features:
	mkdir -p data/processed/crf
	mkdir -p data/processed/crf/$(CRF)
	python src/concat_npy.py --out data/processed/crf/$(CRF)/X_crf.npy --inp data/interim/X_crf_part*
	python src/concat_npy.py --out data/processed/crf/$(CRF)/y_crf.npy --inp data/interim/y_crf_part*
	rm data/interim/X_crf_part*
	rm data/interim/y_crf_part*

crf_train:
	mkdir -p models/$(CRF)
	python src/crf_train.py --data-file data/processed/crf/$(CRF)/X_crf.npy --target-file data/processed/crf/$(CRF)/y_crf.npy --model-filename models/$(CRF)/crf_clf.pkl

crf_segmentation:
	python src/crf_segmentation.py --unary-model-path models/$(UNARY)/unary_clf.pkl --crf-model-path models/$(CRF)/crf_clf.pkl --image-path $(image)

sim_feature_extraction:
	mkdir -p data/processed/sim
	python src/sim_feature_extraction.py --image-dir data/raw/s2s/shop/dresses/test/ --output-dir data/processed/sim/

sim_retrieval:
	python src/sim_retrieval.py --features data/processed/sim/sim_features.pkl --image-query $(image)

pipeline:
	python src/pipeline.py --sim-features data/processed/sim/sim_features.pkl --unary-model-path models/$(UNARY)/unary_clf.pkl --crf-model-path models/$(CRF)/crf_clf.pkl --image-path $(image)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
