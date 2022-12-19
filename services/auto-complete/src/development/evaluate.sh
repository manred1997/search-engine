export SOURCE_DIR=processed-data
echo $SOURCE_DIR
export TARGET_DIR=report

python src/development/evaluate.py --source_dir $SOURCE_DIR \
                                    --target_dir $TARGET_DIR