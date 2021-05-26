### se
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0 --net resnet50 --se Ture ;
python image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --se Ture --net resnet50;
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 1 --net resnet50 --se Ture;
python image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 1  --output_src ckps/source/ --output ckps/target/ --net resnet50 --se Ture;
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 2 --net resnet50 --se Ture ;
python image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 2 --output_src ckps/source/ --output ckps/target/ --net resnet50 --se Ture;
### nonlocal
#python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0 --net resnet50 --nl Ture;
#python image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --net resnet50 --nl Ture;
#python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 1 --net resnet50 --nl Ture;
#python image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 1  --output_src ckps/source/ --output ckps/target/ --net resnet50 --nl Ture;
#python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 2 --net resnet50 --nl Ture;
#python image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 2 --output_src ckps/source/ --output ckps/target/ --net resnet50 --nl Ture;

