import { Point } from './point';


export class Delta {
  start: Point;
  end: Point;

  constructor(start: Point, end: Point) {
    this.start = start;
    this.end = end;
  }

  getDx() {
    return this.end.x - this.start.x;
  }

  getDy() {
    return this.end.y - this.start.y;
  }
}
