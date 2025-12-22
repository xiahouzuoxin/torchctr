import os, sys

sys.path.append('torchctr')
from transformer import FeatureTransformer

if __name__ == '__main__':
    # Test the FeatureTransformer with a sample parquet file
    # generate a sample parquet file

    import polars as pl
    import numpy as np
    num_rows = 10000


    list_cat = [
        f"{a},{b}"
        for a, b in zip(
            np.random.choice(['L1', 'L2', 'L3'], size=num_rows),
            np.random.choice(['L1', 'L2', 'L3'], size=num_rows),
        )
    ]
    list_num = [
        f"{a:.2f},{b:.2f}"
        for a, b in zip(
            np.random.rand(num_rows) * 100,
            np.random.rand(num_rows) * 100,
        )
    ]
    df = pl.DataFrame({
        'cat1': np.random.choice(['A', 'B', 'C', 'D'], size=num_rows),
        'cat2': np.random.choice(['X', 'Y', 'Z'], size=num_rows),
        'num1': np.random.randn(num_rows) * 10 + 50,
        'num2': np.random.rand(num_rows) * 100,
        'list_cat': list_cat,
        'list_num': list_num,
    })

    sample_parquet = 'data/sample_data.parquet'
    df.write_parquet(sample_parquet)   

    print("====> Test fit and transform with DataFrame...")
    ft = FeatureTransformer()
    ft.fit(df)
    df = ft.transform(df)
    print(df.head(10))
    print(ft.get_feat_configs())

    print("====> Test fit_transform with LazyFrame...")
    lf = pl.scan_parquet(sample_parquet)
    ft2 = FeatureTransformer()
    df2 = ft2.fit_transform(lf)
    print( df2.head(10) )
    print(ft2.get_feat_configs())

    print("====> Test fit_transform with parquet file and batch processing...") 
    ft3 = FeatureTransformer()
    ft3.fit_transform(
        sample_parquet,
        output_path='data/transformed_data/',
        batch_size=500,
    )
    print( pl.read_parquet('data/transformed_data').head(10) )
    print(ft3.get_feat_configs())

    # test split function
    print("====> Test split function...")
    lf = pl.scan_parquet(sample_parquet)
    train_df, test_df = ft.split(lf, test_frac=0.2, shuffle=True, random_seed=123, group_col='cat1')
    print("Train DataFrame:")
    print(train_df.collect().shape)
    print("\nTest DataFrame:")
    print(test_df.collect().shape)