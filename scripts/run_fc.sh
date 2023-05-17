
for alpha in 0.05
do
    for seed in 28 37 43 49 7492 4985 2056 2020 2021 2022
    do 
        python3 main_population.py -a $alpha -e 25 -s $seed
    done
done

