conda activate DL

r=1e-4

for alpha in 0. 1e-3 5e-2
do
    python3 conv_main_fptt.py -a $alpha -c 10 10 -lr $r
done
