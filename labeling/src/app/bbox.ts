import { Point } from './point';

export class BBox {
  p2: Point;
  p4: Point;

  constructor(public p1: Point, public p3: Point) {
    this.p2 = new Point(p3.x, p1.y);
    this.p4 = new Point(p1.x, p3.y);
  }

  draw(context: CanvasRenderingContext2D, mode: string) {
    context.globalAlpha = 0.9;

    if (mode == 'Add') {
      context.strokeStyle = '#B900FF';
    } else if (mode === 'Edit') {
      context.strokeStyle = '#FF002B';
    }

    context.beginPath();
    context.moveTo(this.p1.x, this.p1.y);
    context.lineTo(this.p2.x, this.p2.y);
    context.lineTo(this.p3.x, this.p3.y);
    context.lineTo(this.p4.x, this.p4.y);
    context.lineTo(this.p1.x, this.p1.y);
    context.stroke();
  }
}
