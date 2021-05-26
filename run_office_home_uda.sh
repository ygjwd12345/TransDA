
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 100 --s 0;
python image_target.py --cls_par 0.3 --da uda --dset office-home --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --kd Ture;

python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 100 --s 1;
python image_target.py --cls_par 0.3 --da uda --dset office-home --gpu_id 0 --s 1 --output_src ckps/source/ --output ckps/target/ --kd Ture;

python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 100 --s 2;
python image_target.py --cls_par 0.3 --da uda --dset office-home --gpu_id 0 --s 2 --output_src ckps/source/ --output ckps/target/ --kd Ture;

python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 100 --s 3;
python image_target.py --cls_par 0.3 --da uda --dset office-home --gpu_id 0 --s 3 --output_src ckps/source/ --output ckps/target/ --kd Ture;


