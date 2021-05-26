### pda
python image_source.py --trte val --output ckps/source/ --da pda --gpu_id 0 --dset office-home --max_epoch 50 --s 0 --seed  2019;
python image_target.py --cls_par 0.3 --threshold 10 --da pda --dset office-home --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --seed  2019 --kd Ture;

python image_source.py --trte val --output ckps/source/ --da pda --gpu_id 0 --dset office-home --max_epoch 50 --s 1 --seed  2019;
python image_target.py --cls_par 0.3 --threshold 10 --da pda --dset office-home --gpu_id 0 --s 1 --output_src ckps/source/ --output ckps/target/ --seed  2019 --kd Ture;

python image_source.py --trte val --output ckps/source/ --da pda --gpu_id 0 --dset office-home --max_epoch 50 --s 2 --seed  2019;
python image_target.py --cls_par 0.3 --threshold 10 --da pda --dset office-home --gpu_id 0 --s 2 --output_src ckps/source/ --output ckps/target/ --seed  2019 --kd Ture;

python image_source.py --trte val --output ckps/source/ --da pda --gpu_id 0 --dset office-home --max_epoch 50 --s 3 --seed  2019;
python image_target.py --cls_par 0.3 --threshold 10 --da pda --dset office-home --gpu_id 0 --s 3 --output_src ckps/source/ --output ckps/target/ --seed  2019 --kd Ture;


### oda
python image_source.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office-home --max_epoch 50 --s 0 --seed  2019;
python image_target_oda.py --cls_par 0.3 --da oda --dset office-home --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --seed  2019 --kd Ture;

python image_source.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office-home --max_epoch 50 --s 1 --seed  2019;
python image_target_oda.py --cls_par 0.3 --da oda --dset office-home --gpu_id 0 --s 1 --output_src ckps/source/ --output ckps/target/ --seed  2019 --kd Ture;

python image_source.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office-home --max_epoch 50 --s 2 --seed  2019;
python image_target_oda.py --cls_par 0.3 --da oda --dset office-home --gpu_id 0 --s 2 --output_src ckps/source/ --output ckps/target/ --seed  2019 --kd Ture;

python image_source.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office-home --max_epoch 50 --s 3 --seed  2019;
python image_target_oda.py --cls_par 0.3 --da oda --dset office-home --gpu_id 0 --s 3 --output_src ckps/source/ --output ckps/target/ --seed  2019 --kd Ture;
