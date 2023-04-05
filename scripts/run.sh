conda activate DL

for alpha in 0 1e-3 1e-4
do
    python3 conv_main_fptt.py -e $alpha -c 10 10
done
