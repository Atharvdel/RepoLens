"""Tests for ts_js_parser.py (SDD §0, §7 step 3, step 4)."""
from app.indexing.ts_js_parser import (
    parse_ts_js_imports_from_source,
    parse_ts_js_symbols_from_source,
)


def test_parse_ts_symbols_class_and_methods():
    src = b"""
    /** App class documentation */
    export class Application {
        private name: string;

        constructor(name: string) {
            this.name = name;
        }

        /** Run the application */
        public run(): void {
            console.log("Running");
        }
    }
    """
    symbols = parse_ts_js_symbols_from_source(src, filename="app.ts")
    assert len(symbols) == 1
    app_cls = symbols[0]
    assert app_cls.name == "Application"
    assert app_cls.kind == "class"
    assert "App class documentation" in (app_cls.docstring or "")
    assert len(app_cls.methods) >= 2
    method_names = [m.name for m in app_cls.methods]
    assert "constructor" in method_names
    assert "run" in method_names


def test_parse_ts_symbols_functions_and_arrow():
    src = b"""
    export function helperFunction(x: number): number {
        return x * 2;
    }

    export const arrowHandler = (req: any, res: any) => {
        res.send("ok");
    };
    """
    symbols = parse_ts_js_symbols_from_source(src, filename="utils.ts")
    names = [s.name for s in symbols]
    assert "helperFunction" in names
    assert "arrowHandler" in names


def test_parse_ts_imports_es_and_cjs():
    src = b"""
    import React, { useState, useEffect } from 'react';
    import { Button } from './components/Button';
    import type { Config } from '../types';
    const fs = require('fs');
    const path = require('./local_path');
    """
    imports = parse_ts_js_imports_from_source(src, filename="index.tsx")
    modules = [(imp.module, imp.is_relative) for imp in imports]

    assert ("react", False) in modules
    assert ("./components/Button", True) in modules
    assert ("../types", True) in modules
    assert ("fs", False) in modules
    assert ("./local_path", True) in modules
