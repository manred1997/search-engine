python run_split_aol.py --train_start "2006-03-01 00:00:00" --train_end "2006-05-18 00:00:00" \
                        --test_start  "2006-05-25 00:00:00" --test_end  "2006-06-01 00:00:00" \
                        --dev_start "2006-05-18 00:00:00" --dev_end "2006-05-25 00:00:00" \
                        --source_dir "../../../resources/auto-complete/benchmark/AOL-user-ct-collection" \
                        --target_dir "../processed-data"