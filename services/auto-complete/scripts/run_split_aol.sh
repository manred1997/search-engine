export SOURCE_DIR=../../resources/auto-complete/benchmark/AOL-user-ct-collection
export SOURCE_DIR=$PWD"/"$SOURCE_DIR
echo $SOURCE_DIR
export TARGET_DIR=processed-data

python scripts/run_split_aol.py --train_start "2006-03-01 00:00:00" --train_end "2006-05-18 00:00:00" \
                                    --test_start  "2006-05-25 00:00:00" --test_end  "2006-06-01 00:00:00" \
                                    --dev_start "2006-05-18 00:00:00" --dev_end "2006-05-25 00:00:00" \
                                    --source_dir $SOURCE_DIR \
                                    --target_dir $TARGET_DIR