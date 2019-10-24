export class Point {
  constructor(public x: number, public y: number) {
    this.x = x;
    this.y = y;
  }

  highlight(context: CanvasRenderingContext2D, mode: string) {
    context.strokeStyle = '#FF5100';

    context.beginPath();
    context.arc(this.x, this.y, 5, 0, 2 * Math.PI);
    context.stroke();

    context.beginPath();
    context.arc(this.x, this.y, 10, 0, 2 * Math.PI);
    context.stroke();

    context.beginPath();
    context.moveTo(this.x + 10, this.y);
    context.lineTo(this.x + 5, this.y);
    context.stroke();

    context.beginPath();
    context.moveTo(this.x - 10, this.y);
    context.lineTo(this.x - 5, this.y);
    context.stroke();

    context.beginPath();
    context.moveTo(this.x, this.y + 10);
    context.lineTo(this.x, this.y + 5);
    context.stroke();

    context.beginPath();
    context.moveTo(this.x, this.y - 10);
    context.lineTo(this.x, this.y - 5);
    context.stroke();
  }
}
