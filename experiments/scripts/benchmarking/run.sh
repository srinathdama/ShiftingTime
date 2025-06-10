# python train_script_krno_fno2d_v2.py -r 16  --model krno
# python train_script_krno_fno2d_v2.py -r 32 --model krno
# python train_script_krno_fno2d_v2.py -r 64 --model krno
# python train_script_krno_fno2d_v2.py -r 96 --model krno
# python train_script_krno_fno2d_v2.py -r 128 --model krno
# python train_script_krno_fno2d_v2.py -r 160 --model krno


python train_script_fno_3d.py -r 32 --model fno  --fno_modes 17 --nepochs 10
python train_script_fno_3d.py -r 64 --model fno  --fno_modes 33  --nepochs 10
python train_script_fno_3d.py -r 96 --model fno  --fno_modes 49  --nepochs 5
python train_script_fno_3d.py -r 128 --model fno  --fno_modes 65  --nepochs 2
python train_script_fno_3d.py -r 160 --model fno  --fno_modes 81  --nepochs 2
