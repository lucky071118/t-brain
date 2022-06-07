import click


@click.group()
def cli():
    pass


@cli.command(help="Type 'doc2vec' or 'valid' to train the specific model")
@click.argument("model_name")
def train(model_name):

    if model_name == "valid":
        from t_brain.facades import valid_model

        click.echo("Train valid model")
        valid_model.train()
    elif model_name == "doc2vec":
        from t_brain.facades import doc2vec_model

        click.echo("Train doc2vec model")
        doc2vec_model.train()
    else:
        click.echo("The model doenn't exist")


@cli.command(help="Run the server")
def run():
    from t_brain.facades import server

    server.run()


@cli.command(help="Predict a document")
@click.argument("document")
def predict(document):
    from t_brain.facades import valid_model

    click.echo(valid_model.predict([document]))


if __name__ == "__main__":
    cli()
