# Copyright 2024 Gabriel Lindenmaier
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

from pathlib import Path

from sqlitedict import SqliteDict


class SqliteKeyValStore:

    def __init__(self, path_slite_db: str, table: str):
        if not Path(path_slite_db).parent.is_dir:
            raise FileNotFoundError(f"path_sqlite_db {path_slite_db} is not valid file path!")
        self.path_db = path_slite_db
        self.table = table
        self.store = SqliteDict(filename=path_slite_db, tablename=table)

    def write_batch(self, l_keys, l_data, overwrite=True):
        assert len(l_keys) == len(l_data)
        for k, d in zip(l_keys, l_data):
            self.write(key=k, data=d, overwrite=overwrite)
        self.store.commit()

    def write(self, key, data, overwrite=True):
        key = str(key)
        if overwrite:
            self.store[key] = data
        elif key not in self.store:
            self.store[key] = data

    def read_batch(self, l_keys, ignore_na=False):
        l_result = []
        key_in_db = True
        for k in l_keys:
            k = str(k)
            if not ignore_na:
                key_in_db = k in self.store
            if key_in_db:
                l_result.append(self.store[k])
            else:
                raise KeyError(f"{k} not in table {self.table} of SqliteDict "
                               f"database {self.path_db}!")
        return l_result

    def commit(self, blocking=True):
        self.store.commit(blocking=blocking)

    def close(self):
        self.store.commit()
        self.store.close()
