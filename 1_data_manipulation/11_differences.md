Watch out for unexpected differences between between dplyr and pandas. A few examples are described below.

## Operations on Grouped Data

With dplyr, the command

```r
inventory %>% group_by(shop) %>% head(1)
```

returns just one row:

| shop      | game     | qty | aisle | price |
|-----------|----------|-----|-------|-------|
| Dicey     | Monopoly | 7   | 3     | 17.99 |


but with pandas, the command

```python
inventory.groupby('shop').head(1)
```

returns two rows, one for each group:

| shop      | game     | qty | aisle | price |
|-----------|----------|-----|-------|-------|
| Dicey     | Monopoly | 7   | 3     | 17.99 |
| Board 'Em | Monopoly | 11  | 2     | 25.00 |

## Value Semantics versus Reference Semantics

R and dplyr always use _value semantics_ when assigning data frames or passing data frames as arguments to functions. However, Python and pandas sometimes use _reference semantics_. For example, if you assign the pandas DataFrame in the variable named `games` to a new variable named `boardgames`:

```python
boardgames = games
```

then you make an in-place modification to `boardgames`

```python
boardgames.iloc[2,1] = 'Cluedo'
```

you might be surprised to see that `games` has also changed, and the command

```python
games.iloc[2,1]
```

returns

```
'Cluedo'
```

This happens because `boardgames` is a _pointer_ to `games`, not a _copy_ of games. To make a _copy_ of `games`, use

```python
boardgames = games.copy()
```
