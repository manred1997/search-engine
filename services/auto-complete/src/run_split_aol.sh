python run_split_aol.py --train_start "2006-03-01 00:00:00" --train_end "2006-05-18 00:00:00" \
                        --valid_start "2006-05-18 00:00:00" --valid_end "2006-05-25 00:00:00" \
                        --test_start  "2006-05-25 00:00:00" --test_end  "2006-06-01 00:00:00" \
                        --aol_benchmark_dir "../../../resources/auto-complete/benchmark/AOL-user-ct-collection" \
                        --target_dir "../processed-data"