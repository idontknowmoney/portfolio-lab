from portfolio_lab import main


def test_main_prints_greeting(capsys):
    main()
    assert "portfolio-lab" in capsys.readouterr().out
