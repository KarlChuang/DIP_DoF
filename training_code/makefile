train_%:
	python train.py \
	--num_epochs 1200 \
	--arch $* \
	--lr 0.001 \
	--train_batch_size 128 \
	--print_freq 10 \
	--src_label_path ../data/train/groundTruth7000.csv \
	--src_data_path ../data/train/imageBlur \
	--lr_gamma 0.5 \
	--lr_decay_step 20
# --pretrained \
# --source_dir ./experiment/train/checkpoint/ \
# --source_file model_best.pt

continue_%:
	python train.py \
	--num_epochs 1200 \
	--arch $* \
	--lr 0.0003 \
	--train_batch_size 128 \
	--print_freq 10 \
	--src_label_path ../data/train/groundTruth7000.csv \
	--src_data_path ../data/train/imageBlur \
	--lr_gamma 0.5 \
	--lr_decay_step 20 \
	--resume ../result/$*/checkpoint/model_96.pt

setup_%:
	-mkdir ../result
	-mkdir ../result/$*
	cp -R ./experiment/train/checkpoint ../result/$*/
	cp ./experiment/train/config.txt ../result/$*/
	cp ./experiment/train/learning_curve.csv ../result/$*/

test_%:
	python train.py --test_only --pretrained --source_dir ../result/$*/checkpoint/ --output_file ../result/$*/output.csv --arch $* --final_test_batch_size 16

plot_%:
	python plot.py \
	--src_label_path ../data/train/groundTruth.csv \
	--src_data_path ../data/train/imageBlur \
	--source_dir ../result/$*/checkpoint/ \
	--source_file model_best.pt \
	--test_only \
	--plot_csv ../result/$*/learning_curve.csv 

eval_%:
	python train.py --inference_only --pretrained --source_dir ./experiment/train/checkpoint/ --output_file ./experiment/train/output.csv --arch $*

upload_%:
	kaggle competitions submit -c ntuds-2021-digit-recognition -f ../result/$*/output.csv -m "$*"

clean:
	rm -rf experiment