
for alpha in 0.0
do
    for seed in 999
    do 
        python3 main_population.py -a $alpha -e 25 -s $seed
    done
done

