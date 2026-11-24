.PHONY: up down logs test test-local reset data eval eval-detector eval-sam eval-effort eval-failures eval-quick eval-sim eval-al

up:            ## build and start everything (http://localhost:8080)
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f backend

test:          ## run backend tests inside the container
	docker compose run --rm --no-deps -e AI_ENABLED=0 backend python -m pytest -q tests

test-local:    ## run the dependency-light tests on the host
	cd backend && python -m unittest discover -s tests -v

reset:         ## delete all data (database + uploaded images)
	docker compose down -v

RAW ?= data/raw/detection/train

data:          ## prepare MinneApple inside Docker (dataset extracted to data/raw)
	docker compose run --rm backend python scripts/minneapple_to_coco.py --images $(RAW)/images --masks $(RAW)/masks --out data/minneapple
	docker compose run --rm backend python scripts/make_crops.py --images $(RAW)/images --masks $(RAW)/masks --split study
	docker compose run --rm backend python scripts/make_crops.py --images $(RAW)/images --masks $(RAW)/masks --split pool --min-apples 1 --max-apples 1000
	docker compose run --rm backend python scripts/make_crops.py --images $(RAW)/images --masks $(RAW)/masks --split test --min-apples 1 --max-apples 1000
	docker compose run --rm backend python scripts/show_gt.py --coco data/minneapple/coco_crops_study.json --images data/minneapple/crops_study/images --out data/gt_check.png

RUN = docker compose run --rm backend python

eval-detector: ## learning curve: pretrained vs fine-tuned on 25/50/100/200 crops (slow on CPU)
	$(RUN) eval/detector_curve.py --sizes 0 25 50 100 200

eval-quick:    ## fast smoke run of the model evaluation (few crops) to check everything works
	$(RUN) eval/detector_curve.py --sizes 0 10 --epochs 2 --max-test 20
	$(RUN) eval/sam_noc.py --max-objects 30

eval-sam:      ## SAM NoC@85/90, mIoU per click, box-prompt IoU
	$(RUN) eval/sam_noc.py --max-objects 400

eval-effort:   ## manual vs assisted from the app logs (see eval/README.md)
	$(RUN) eval/effort.py --logs data/logs.json --export data/study_export.json --gt data/minneapple/coco_crops_study.json

eval-failures: ## failure figures (needs eval-sam and eval-detector first)
	$(RUN) eval/failure_gallery.py --detections eval/results/detections_n100.json

eval-sim:      ## simulated annotator: manual vs SAM vs detector+SAM (pretrained and fine-tuned)
	$(RUN) eval/simulate_annotator.py --detections eval/results/detections_n0.json eval/results/detections_n100.json $(if $(wildcard data/logs.json),--logs data/logs.json,)

eval-al:       ## active learning: uncertainty vs random sampling (slow, GPU recommended)
	$(RUN) eval/active_learning.py --rounds 4

eval: eval-detector eval-sam eval-failures eval-sim
