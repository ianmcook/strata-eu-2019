# Copyright 2018 Cloudera, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # Creating new columns in a pandas DataFrame

# Import modules and read games data
import numpy as np
import pandas as pd
games = pd.read_table('data/games/games.csv', sep=',')
games


# Use the DataFrame method `assign` method to return the
# dataframe with a new column added to it.

# The new column can contain a scalar value (repeated in 
# every row):
games.assign(tax_percent = 0.08875)

# Or, the new column can be calculated using an
# expression that uses the values in other columns. To
# do this, you need to reference the original DataFrame
# or use a lambda:
games \
  .assign(
    price_with_tax = round(
      games.list_price * 1.08875, 2
    )
  )

games \
  .assign(
    price_with_tax = lambda x:
      round(x.list_price * 1.08875, 2)
  )

# The latter option is better for chaining

# You can also use the quoted column name in square
# brackets. This is less concise but safer, because 
# Python can interpret the dot notation to refer to
# an attribute or method of the DataFrame class
# instead of a column of the DataFrame
games.assign(under_ten_dollars = games.list_price < 10)
games.assign(under_ten_dollars = games['list_price'] < 10)


# You can create multiple columns with one call to the
# `assign` method
games \
  .assign(
    tax_percent = 0.08875,
    price_with_tax = lambda x:
      round(x.list_price * 1.08875, 2)
  )

# But you can't reference columns that you created in an
# `assign` in other expressions later in the same 
# `assign`; for that, use multiple `assign`s
games \
  .assign(
    tax_percent = 0.08875
  ) \
  .assign(
    price_with_tax = lambda x:
      round(x.list_price * (1 + x.tax_percent), 2)
  )


# ## Replacing columns

# You can replace existing columns the same way you make
# new columns
games. \
  assign(
    name = lambda x: x.name.str.upper(),
    inventor = lambda x: x.inventor.str.lower()
  )


# ## Renaming columns
  
# To return a DataFrame with one or more columns renamed,
# use the `rename` method. For the `columns` argument,
# pass a dictionary in the form  `{'old_name':'new_name'}`
games.rename(columns = {'id':'game_id'})
games.rename(
  columns = {'id':'game_id', 'list_price':'price'}
)


# ## Removing columns

# To return a DataFrame with one or more columns removed,
# use the `drop` method, with `axis=1`
games.drop(['inventor', 'min_age'], axis=1)


# ## Replacing missing values

# Load the inventory data (since the games data has no
# missing values)
inventory = pd.read_table('data/inventory/data.txt')
inventory

# Use the `fillna` method
inventory.assign(price = lambda x : x.price.fillna(9.00))
