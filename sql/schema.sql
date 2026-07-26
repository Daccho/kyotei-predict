-- kyotei-ml schema (Phase 1)
--
-- Design notes
--   * Natural composite key (race_date, stadium_id, race_no) identifies a race.
--     No surrogate key: the load path never needs a lookup round-trip, and the
--     key is identical in Postgres and in the polars/parquet mirror.
--   * entries.lane (枠番) and entries.course (実際の進入コース) are SEPARATE
--     columns. They coincide only when nobody changes course before the start;
--     course is the dominant feature (SPEC §2 A) and lane is the draw. Collapsing
--     them would silently destroy the single most predictive input.
--   * Columns that only the B file knows (pre-race) and columns only the K file
--     knows (post-race) live in the same row but are loaded by different passes,
--     so almost everything is NULLable. A NULL means "not yet ingested or not
--     published", never "zero".
--   * ingest_log records per-file parse outcomes so Phase 1 can report parse
--     success rate BY YEAR and detect the layout changes SPEC §3.2 warns about.

BEGIN;

-- ---------------------------------------------------------------------------
-- Reference: the 24 stadiums
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stadiums (
    stadium_id  SMALLINT PRIMARY KEY CHECK (stadium_id BETWEEN 1 AND 24),
    name        TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- races: one row per race
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS races (
    race_date        DATE     NOT NULL,
    stadium_id       SMALLINT NOT NULL REFERENCES stadiums (stadium_id),
    race_no          SMALLINT NOT NULL CHECK (race_no BETWEEN 1 AND 12),

    -- from B (番組表)
    title            TEXT,             -- レース名
    grade_class      TEXT,             -- 一般/G3/G2/G1/SG など
    distance_m       SMALLINT CHECK (distance_m IS NULL OR distance_m BETWEEN 800 AND 3000),
    deadline_time    TIME,             -- 締切予定時刻: the as-of cutoff for features
    series_day       TEXT,             -- 節間の何日目か（初日/2日目/最終日 …）

    -- from K (競走成績): conditions are only final near the deadline
    weather          TEXT,             -- 晴/曇り/雨/雪/霧
    wind_direction   SMALLINT CHECK (wind_direction IS NULL OR wind_direction BETWEEN 1 AND 16),
    wind_speed_m     SMALLINT CHECK (wind_speed_m IS NULL OR wind_speed_m BETWEEN 0 AND 30),
    wave_height_cm   SMALLINT CHECK (wave_height_cm IS NULL OR wave_height_cm BETWEEN 0 AND 100),
    air_temp_c       REAL,
    water_temp_c     REAL,
    decision         TEXT,             -- 決まり手: 逃げ/差し/まくり/まくり差し/抜き/恵まれ

    -- provenance
    has_b            BOOLEAN NOT NULL DEFAULT FALSE,
    has_k            BOOLEAN NOT NULL DEFAULT FALSE,

    PRIMARY KEY (race_date, stadium_id, race_no)
);

CREATE INDEX IF NOT EXISTS races_date_idx ON races (race_date);
CREATE INDEX IF NOT EXISTS races_stadium_idx ON races (stadium_id);

-- ---------------------------------------------------------------------------
-- entries: one row per boat per race (expected: exactly 6)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entries (
    race_date            DATE     NOT NULL,
    stadium_id           SMALLINT NOT NULL,
    race_no              SMALLINT NOT NULL,

    lane                 SMALLINT NOT NULL CHECK (lane BETWEEN 1 AND 6),   -- 枠番 (the draw)
    course               SMALLINT CHECK (course IS NULL OR course BETWEEN 1 AND 6), -- 実際の進入コース

    -- racer identity / class (from B)
    racer_id             INTEGER,      -- 登番
    racer_name           TEXT,
    grade                TEXT CHECK (grade IS NULL OR grade IN ('A1', 'A2', 'B1', 'B2')),
    branch               TEXT,         -- 支部
    birthplace           TEXT,
    age                  SMALLINT,
    weight_kg            REAL,

    -- rolling form published on the race card. NOTE: these are the *published*
    -- period figures, not leak-free rolling features. features.py must not use
    -- them without checking the publication cutoff.
    national_win_rate    REAL,
    national_top2_rate   REAL,
    local_win_rate       REAL,         -- 当地勝率
    local_top2_rate      REAL,         -- 当地2連率
    motor_no             SMALLINT,
    motor_top2_rate      REAL,         -- raw rate; shrink toward the mean in features.py
    boat_no              SMALLINT,
    boat_top2_rate       REAL,

    -- pre-race measurements (finalised only shortly before the start)
    exhibition_time      REAL,         -- 展示タイム
    exhibition_start     REAL,         -- スタート展示ST
    tilt                 REAL,         -- チルト角度
    weight_adjust_kg     REAL,         -- 当日体重調整

    -- outcome (from K)
    finish_position      SMALLINT CHECK (finish_position IS NULL OR finish_position BETWEEN 1 AND 6),
    finish_status        TEXT,         -- F/L/S0/S1/S2/転覆/落水/失格/不完走/欠場 …
    start_timing         REAL,         -- ST. negative = フライング
    race_time_sec        REAL,

    PRIMARY KEY (race_date, stadium_id, race_no, lane),
    FOREIGN KEY (race_date, stadium_id, race_no)
        REFERENCES races (race_date, stadium_id, race_no) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS entries_racer_date_idx ON entries (racer_id, race_date);
CREATE INDEX IF NOT EXISTS entries_motor_idx ON entries (stadium_id, motor_no, race_date);
CREATE INDEX IF NOT EXISTS entries_date_idx ON entries (race_date);

-- A boat's actual course must be unique within a race: two boats cannot occupy
-- the same starting course. Enforced as a partial unique index because course
-- is NULL until the K file lands.
CREATE UNIQUE INDEX IF NOT EXISTS entries_course_unique_idx
    ON entries (race_date, stadium_id, race_no, course)
    WHERE course IS NOT NULL;

-- ---------------------------------------------------------------------------
-- payouts: real dividends, the basis of the backtest (SPEC §2.5)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payouts (
    race_date     DATE     NOT NULL,
    stadium_id    SMALLINT NOT NULL,
    race_no       SMALLINT NOT NULL,
    bet_type      TEXT     NOT NULL CHECK (bet_type IN
                     ('trifecta',    -- 3連単
                      'trio',        -- 3連複
                      'exacta',      -- 2連単
                      'quinella',    -- 2連複
                      'win',         -- 単勝
                      'place',       -- 複勝
                      'wide')),      -- 拡連複
    combination   TEXT     NOT NULL,  -- canonical, e.g. '1-2-3' (ordered) or '1=2=3' (unordered)
    payout_yen    INTEGER  CHECK (payout_yen IS NULL OR payout_yen >= 0),  -- per 100 yen stake
    popularity    SMALLINT,           -- 人気

    PRIMARY KEY (race_date, stadium_id, race_no, bet_type, combination),
    FOREIGN KEY (race_date, stadium_id, race_no)
        REFERENCES races (race_date, stadium_id, race_no) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS payouts_type_idx ON payouts (bet_type, race_date);

-- ---------------------------------------------------------------------------
-- racers: half-yearly published records (fan{YYMM})
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS racers (
    racer_id          INTEGER  NOT NULL,
    period_year       SMALLINT NOT NULL,
    period_half       SMALLINT NOT NULL CHECK (period_half IN (1, 2)),
    -- The date this record becomes usable without leaking. A period's figures
    -- describe races BEFORE it is published, so features must join as-of this
    -- date, never on the period the race falls in.
    effective_from    DATE     NOT NULL,

    name              TEXT,
    grade             TEXT,
    branch            TEXT,
    birthplace        TEXT,
    age               SMALLINT,
    weight_kg         REAL,
    win_rate          REAL,
    top2_rate         REAL,
    top3_rate         REAL,
    avg_start_timing  REAL,     -- 平均ST
    starts            INTEGER,  -- 出走回数: the denominator for shrinkage
    firsts            INTEGER,
    seconds           INTEGER,
    -- 決まり手傾向 (SPEC §2 C)
    nige_count        INTEGER,  -- 逃げ
    sashi_count       INTEGER,  -- 差し
    makuri_count      INTEGER,  -- まくり

    PRIMARY KEY (racer_id, period_year, period_half)
);

CREATE INDEX IF NOT EXISTS racers_effective_idx ON racers (racer_id, effective_from);

-- ---------------------------------------------------------------------------
-- ingest_log: per-file parse outcome, for the by-year success rates
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_log (
    kind            TEXT NOT NULL CHECK (kind IN ('B', 'K', 'fan')),
    source_name     TEXT NOT NULL,     -- e.g. 'b200714' / 'fan1410'
    race_date       DATE,              -- NULL for fan files
    races_parsed    INTEGER NOT NULL DEFAULT 0,
    entries_parsed  INTEGER NOT NULL DEFAULT 0,
    payouts_parsed  INTEGER NOT NULL DEFAULT 0,
    error_count     INTEGER NOT NULL DEFAULT 0,
    first_error     TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (kind, source_name)
);

COMMIT;

-- ===========================================================================
-- Validation views. These ARE the Phase 1 completion evidence; they are
-- queried by `python -m kyotei.load --validate`, not eyeballed.
-- ===========================================================================

-- Races whose entry count is not exactly 6.
CREATE OR REPLACE VIEW v_bad_entry_counts AS
SELECT r.race_date,
       r.stadium_id,
       r.race_no,
       count(e.lane) AS entry_count
FROM races r
LEFT JOIN entries e USING (race_date, stadium_id, race_no)
GROUP BY r.race_date, r.stadium_id, r.race_no
HAVING count(e.lane) <> 6;

-- Per-year counts, plus how often course actually differs from lane. If
-- course_differs_pct is 0 the K parser is copying lane into course -- the exact
-- silent bug the separate columns exist to prevent.
CREATE OR REPLACE VIEW v_yearly_summary AS
SELECT extract(YEAR FROM r.race_date)::INT              AS year,
       count(DISTINCT (r.race_date, r.stadium_id, r.race_no)) AS races,
       count(e.lane)                                    AS entries,
       count(e.course)                                  AS entries_with_course,
       count(e.finish_position)                         AS entries_with_finish,
       round(100.0 * count(*) FILTER (WHERE e.course IS NOT NULL AND e.course <> e.lane)
             / NULLIF(count(e.course), 0), 2)           AS course_differs_pct,
       round(avg(e.course) FILTER (WHERE e.finish_position = 1), 3) AS avg_winning_course
FROM races r
LEFT JOIN entries e USING (race_date, stadium_id, race_no)
GROUP BY 1
ORDER BY 1;

-- Sanity: share of races won from course 1. Expect roughly 0.50-0.58.
CREATE OR REPLACE VIEW v_course1_winrate AS
SELECT extract(YEAR FROM race_date)::INT AS year,
       count(*)                          AS races_with_winner,
       round(avg((course = 1)::INT)::NUMERIC, 4) AS course1_win_rate
FROM entries
WHERE finish_position = 1 AND course IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- Races that have results but no trifecta dividend (breaks the backtest).
CREATE OR REPLACE VIEW v_missing_trifecta AS
SELECT r.race_date, r.stadium_id, r.race_no
FROM races r
WHERE r.has_k
  AND NOT EXISTS (
      SELECT 1 FROM payouts p
      WHERE p.race_date = r.race_date
        AND p.stadium_id = r.stadium_id
        AND p.race_no = r.race_no
        AND p.bet_type = 'trifecta'
  );
