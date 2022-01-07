train_%:
	python train.py \
	--num_epochs 120 \
	--arch $* \
	--lr 0.005 \
	--train_batch_size 128 \
	--print_freq 100

continue_%:
	python train.py \
	--num_epochs 120 \
	--arch $* \
	--lr 0.00005 \
	--train_batch_size 128 \
	--print_freq 100 \
	--pretrained \
	--source_dir ../result/mobile2/checkpoint/ \
	--source_file model_47.pt

setup_%:
	-mkdir ../result
	-mkdir ../result/$*
	cp -R ./experiment/train/checkpoint ../result/$*/
	cp ./experiment/train/config.txt ../result/$*/
	cp ./experiment/train/learning_curve.csv ../result/$*/

test_%:
	python train.py --test_only --pretrained --source_dir ../result/$*/checkpoint/ --output_file ../result/$*/output.csv --arch $* --final_test_batch_size 16

plot_%:
	python plot.py --source_dir ../result/$*/checkpoint/ --arch $* --test_only --plot_csv ../result/$*/learning_curve.csv --num_epochs 120

eval_%:
	python train.py --inference_only --pretrained --source_dir ./experiment/train/checkpoint/ --output_file ./experiment/train/output.csv --arch $*

upload_%:
	kaggle competitions submit -c ntuds-2021-digit-recognition -f ../result/$*/output.csv -m "$*"

clean:
	rm -rf experiment