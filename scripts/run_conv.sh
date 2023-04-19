
r=1e-3
convadp=True

for alpha in 1. 0
do
    python3 conv_main_fptt.py -a $alpha -c 2 -lr $r -k 16 -ro 50 -dt 1 -e 10 -ca $convadp
done

